"""Train the tiny GPT on english -> regex pairs.

    python src/train.py --name m3l --dim 192 --layers 4 --heads 4 --lr 2.5e-3 \
        --dropout 0.05 --ema 0.999 --steps 60000 --eval_every 10000
"""
import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

import evaluate
import infer
import tokenizer as tok
from model import GPT, Config

ROOT = Path(__file__).parents[1]


def load(split, max_len):
    rows = [json.loads(l) for l in open(ROOT / f"data/{split}.jsonl")]
    en = tok.english_batch([r["en"] for r in rows])
    keep = [i for i, e in enumerate(en) if len(e) + len(r := rows[i]["re"].encode()) + 2 <= max_len]
    rows = [rows[i] for i in keep]
    return rows, [(rows[i]["en"], tok.encode(rows[i]["re"]), len(en[k])) for i, k in enumerate(keep)]


def augment(s, rng):
    """Cheap robustness noise: casing and trailing punctuation."""
    r = rng.random()
    if r < 0.12:
        return s.lower()
    if r < 0.18:
        return s.rstrip(".!?")
    if r < 0.21:
        return s[:1].lower() + s[1:]
    return s


def batches(data, bsz, rng, aug=False):
    """Length-bucketed batches, shuffled. data: (en, regex ids, en len) -> (x, y, w)."""
    idx = sorted(range(len(data)), key=lambda i: (data[i][2] + len(data[i][1]), rng.random()))
    chunks = [idx[i:i + bsz] for i in range(0, len(idx), bsz)]
    rng.shuffle(chunks)
    for ch in chunks:
        ens = tok.english_batch([augment(data[i][0], rng) if aug else data[i][0] for i in ch])
        seqs = [(e + [tok.SEP] + data[i][1] + [tok.EOS], len(e)) for e, i in zip(ens, ch)]
        T = max(len(t) for t, _ in seqs)
        T = (T + 7) // 8 * 8 + 1  # bucket shapes -> few recompiles, little padding
        x = np.full((len(ch), T), tok.PAD, np.int32)
        w = np.zeros((len(ch), T - 1), np.float32)
        for r, (t, sep) in enumerate(seqs):
            x[r, :len(t)] = t
            w[r, sep:len(t) - 1] = 1.0  # predict regex + EOS
            w[r, :sep] = -1.0  # marker: english positions (for optional aux loss)
        yield x[:, :-1], x[:, 1:], w


def prefix_mask(w):
    """(B,1,T,T) additive mask: causal, plus full attention within the prompt."""
    B, T = w.shape
    n = (w < 0).sum(1) + 1  # prompt length incl. SEP
    i = np.arange(T)
    pre = (i[None, :] < n[:, None])
    ok = (i[None, :, None] >= i[None, None, :]) | (pre[:, :, None] & pre[:, None, :])
    return np.where(ok, 0.0, -1e9).astype(np.float32)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--steps", type=int, default=0, help="fixed step budget (overrides --minutes)")
    ap.add_argument("--epochs", type=float, default=100)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of LM loss on the english prefix")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_n", type=int, default=600)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 mixed precision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", type=int, default=0, help="prefix-LM attention over the prompt")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights for eval/save, e.g. 0.999")
    ap.add_argument("--aug", type=int, default=1, help="casing/punctuation noise on english")
    ap.add_argument("--init", default=None, help="ckpt dir to warm-start from")
    a = ap.parse_args()

    mx.random.seed(a.seed)
    rng = random.Random(a.seed)
    _, train = load("train", a.max_len)
    val_rows, val = load("val", a.max_len)
    ev_idx = random.Random(1).sample(range(len(val_rows)), min(a.eval_n, len(val_rows)))
    ev_rows = [val_rows[i] for i in ev_idx]

    cfg = Config(vocab=tok.vocab(), dim=a.dim, layers=a.layers, heads=a.heads, dropout=a.dropout, max_len=a.max_len,
                 compute="float32" if a.fp32 else "bfloat16", prefix=bool(a.prefix))
    model = GPT(cfg)
    if a.init:
        model.load_weights(str(Path(a.init) / "model.safetensors"))
    nparams = model.n_params()
    ntok = sum(n + len(r) + 2 for _, r, n in train)
    steps_per_epoch = math.ceil(len(train) / a.bsz)
    total = int(steps_per_epoch * a.epochs)
    print(f"{a.name}: {nparams / 1e6:.2f}M params, {len(train)} train pairs ({ntok / 1e6:.1f}M tok), {steps_per_epoch} steps/epoch")

    # schedule length is time-based: estimate after warmup, fixed afterwards
    sched_total = [a.steps or total]

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        p = min(1.0, (step - a.warmup) / max(1, sched_total[0] - a.warmup))
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.95], weight_decay=a.wd)

    def loss_fn(model, x, y, w, mask="causal"):
        logits = model(x, None, mask if a.prefix else "causal").astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, y, reduction="none")
        tgt = (w > 0).astype(mx.float32)
        loss = (ce * tgt).sum() / tgt.sum()
        if a.aux > 0:
            eng = (w < 0).astype(mx.float32)
            loss = loss + a.aux * (ce * eng).sum() / mx.maximum(eng.sum(), 1)
        return loss, tgt.sum()

    _dummy = mx.zeros((1,))

    def mask_of(w):
        return mx.array(prefix_mask(w)) if a.prefix else _dummy

    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, w, mask):
        (loss, n), g = nn.value_and_grad(model, loss_fn)(model, x, y, w, mask)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    ema = [tree_map(lambda p: p, model.trainable_parameters())] if a.ema else None

    class swap:  # evaluate/save with EMA weights, then restore
        def __enter__(self):
            if ema:
                self.raw = model.trainable_parameters()
                model.update(ema[0])

        def __exit__(self, *_):
            if ema:
                model.update(self.raw)

    def val_loss():
        model.eval()
        tot = cnt = 0.0
        for x, y, w in batches(val[:3000], 256, random.Random(0)):
            l, n = loss_fn(model, mx.array(x), mx.array(y), mx.array(w), mask_of(w))
            tot += l.item() * n.item()
            cnt += n.item()
        model.train()
        return tot / cnt

    def val_decode():
        model.eval()
        preds = []
        for i in range(0, len(ev_rows), 200):
            preds += infer.greedy(model, [tok.prompt(r["en"]) for r in ev_rows[i:i + 200]])
        model.train()
        return evaluate.score(preds, ev_rows) | {"styles": evaluate.by_style(preds, ev_rows)}

    out = ROOT / "ckpt" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.dict()))
    (out / "args.json").write_text(json.dumps(vars(a)))
    log = open(out / "log.jsonl", "w")

    model.train()
    t0, it, best, ep, losses = time.time(), 0, -1.0, 0, []
    deadline = t0 + (1e9 if a.steps else a.minutes * 60)
    stop = False
    while not stop and ep < a.epochs:
        for x, y, w in batches(train, a.bsz, rng, aug=a.aug):
            opt.learning_rate = lr_at(it)
            loss = step(mx.array(x), mx.array(y), mx.array(w), mask_of(w))
            mx.async_eval(loss, state)  # overlap next batch prep with GPU
            if losses:
                mx.eval(losses[-1])  # bound the queue to one step ahead
            losses.append(loss)
            if ema and it % 4 == 0:
                d = a.ema ** 4
                ema[0] = tree_map(lambda e, p: e * d + p * (1 - d), ema[0], model.trainable_parameters())
                mx.async_eval(ema[0])
            if len(losses) > 400:
                losses = [float(l) for l in losses[-200:]]
            it += 1
            if it == a.warmup + 50 and not a.steps:  # fit schedule to the time budget
                rate = it / (time.time() - t0)
                sched_total[0] = min(total, int(rate * a.minutes * 60 * 0.93))
                print(f"  {rate:.1f} it/s -> schedule {sched_total[0]} steps ({sched_total[0] / steps_per_epoch:.1f} ep)")
            last = time.time() > deadline or it >= sched_total[0]
            if it % a.eval_every == 0 or last:
                with swap():
                    m = val_decode() | {"step": it, "epoch": round(it / steps_per_epoch, 2), "val": val_loss(),
                                        "train": float(np.mean([float(l) for l in losses[-200:]])),
                                        "lr": lr_at(it), "min": round((time.time() - t0) / 60, 1)}
                    if m["equiv"] > best:
                        best = m["equiv"]
                        model.save_weights(str(out / "model.safetensors"))
                log.write(json.dumps(m) + "\n")
                log.flush()
                print(f"  step {it} ep {m['epoch']} train {m['train']:.4f} val {m['val']:.4f} exact {m['exact']:.3f} "
                      f"equiv {m['equiv']:.3f} comp {m['compiles']:.3f} [{m['min']}m]", flush=True)
            if last:
                stop = True
                break
        ep += 1
    print(f"{a.name}: best equiv {best:.3f}, {nparams / 1e6:.2f}M params, {it} steps, {(time.time() - t0) / 60:.1f} min")
    (out / "result.json").write_text(json.dumps({"best_equiv": best, "params": nparams, "steps": it}))


if __name__ == "__main__":
    main()
