"""Decoding (batched greedy, beam search, example-aware ranking) + CLI.

    python src/infer.py "five digit zip code, optionally dash and four more"
    python src/infer.py -k 5 "an email address"     # top 5 candidates
    python src/infer.py --bits 8 "a uuid"           # 8-bit quantized
    python src/infer.py                             # interactive
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

import engine
import grammar
import tokenizer as tok
from model import GPT, Config

ROOT = Path(__file__).parents[1]
HERE = Path(__file__).parent
DEFAULT = HERE if (HERE / "model.safetensors").exists() else ROOT / "ckpt/best"  # flat release or repo
NEG = -1e9


def load(path, bits=0):
    """bits=0: as trained; 8 or 4: quantized linears (smaller, faster)."""
    from model import quantize
    path = Path(path)
    c = Config(**json.loads((path / "config.json").read_text()))
    m = GPT(c)
    m.load_weights(str(path / "model.safetensors"))
    if bits:
        quantize(m, bits)
    m.eval()
    mx.eval(m.parameters())
    return m


def greedy(model, prompts, max_new=96):
    """Batched greedy decode with left padding. prompts: list[list[int]]."""
    B, L = len(prompts), max(map(len, prompts))
    pad = mx.array([L - len(p) for p in prompts])
    x = mx.array([[tok.PAD] * (L - len(p)) + p for p in prompts])
    pos = mx.maximum(mx.arange(L)[None] - pad[:, None], 0)
    valid = mx.arange(L)[None] >= pad[:, None]  # B,L
    causal = mx.tril(mx.ones((L, L), dtype=mx.bool_))
    if model.c.prefix:  # whole prompt is prefix: attend to every real token
        causal = mx.ones((L, L), dtype=mx.bool_)
    mask = mx.where(causal[None, None] & valid[:, None, None, :], 0.0, NEG)
    cache = [[None, None] for _ in model.blocks]
    logits = model(x, pos, mask, cache)[:, -1]
    out = [[] for _ in range(B)]
    done = [False] * B
    cur = mx.array([len(p) for p in prompts])
    kmask = mx.where(valid, 0.0, NEG)
    for _ in range(max_new):
        nxt = mx.argmax(logits, axis=-1)
        ids = nxt.tolist()
        for i, t in enumerate(ids):
            if not done[i]:
                if t == tok.EOS:
                    done[i] = True
                else:
                    out[i].append(t)
        if all(done):
            break
        kmask = mx.concatenate([kmask, mx.zeros((B, 1))], axis=1)
        logits = model(nxt[:, None], cur[:, None], kmask[:, None, None, :], cache)[:, -1]
        cur = cur + 1
    return [tok.decode(o) for o in out]


def beam(model, prompt, k=8, max_new=96, alpha=0.6):
    """Beam search for one prompt -> list of (regex, score), best first."""
    x = mx.array([prompt])
    cache = [[None, None] for _ in model.blocks]
    logp = mx.log(mx.softmax(model(x, None, None if model.c.prefix else "causal", cache)[:, -1].astype(mx.float32), axis=-1))
    T = len(prompt)
    beams = [([], 0.0)]
    finished = []
    for step in range(max_new):
        lp = logp + mx.array([b[1] for b in beams])[:, None]
        flat = lp.reshape(-1)
        top = mx.argsort(-flat)[: 2 * k].tolist()
        V = logp.shape[-1]
        nb, src = [], []
        fl = flat.tolist() if len(top) else []
        for idx in top:
            bi, t = divmod(idx, V)
            s = fl[idx]
            seq = beams[bi][0]
            if t == tok.EOS:
                finished.append((seq, s / ((5 + len(seq)) / 6) ** alpha))
            elif t >= tok.OFFSET:
                nb.append((seq + [t], s))
                src.append(bi)
            if len(nb) == k:
                break
        if not nb or (len(finished) >= k and max(f[1] for f in finished) > max(s for _, s in nb) / ((5 + step + 1) / 6) ** alpha):
            break
        idx = mx.array(src)
        for c in cache:
            c[0], c[1] = c[0][idx], c[1][idx]
        beams = nb
        x = mx.array([[s[-1]] for s, _ in beams])
        pos = mx.full((len(beams), 1), T + step)
        logp = mx.log(mx.softmax(model(x, pos, None, cache)[:, -1].astype(mx.float32), axis=-1))
    finished.sort(key=lambda f: -f[1])
    seen, res = set(), []
    for seq, s in finished:
        r = tok.decode(seq)
        if r not in seen:
            seen.add(r)
            res.append((r, s))
    return res


def valid(regex):
    try:
        p, f = engine.parse_literal(regex)
    except ValueError:
        return False
    return engine.compiles(p, f)


def best(model, en, k=8, use_hints=True):
    """Beam search, then rank: compiles > matches examples in the request > in scope > model score."""
    import hints
    cands = beam(model, tok.prompt(en), k=k)
    ex = hints.examples(en) if use_hints else []

    def rank(c):
        r = c[0]
        try:
            p, f = engine.parse_literal(r)
            ok = engine.compiles(p, f)
        except ValueError:
            return (1, 0, 1, -c[1])
        hit = sum(engine.test(p, f, ex)) if ok and ex else 0
        scope = ok and grammar.in_scope(p, f)
        return (not ok, -hit, not scope, -c[1])
    return sorted(cands, key=rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--ckpt", default=DEFAULT)
    ap.add_argument("-k", type=int, default=1, help="show top-k candidates")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--bits", type=int, default=0, help="quantize linears: 8 or 4")
    a = ap.parse_args()
    model = load(a.ckpt, a.bits)

    def run(en):
        for r, s in best(model, en, k=max(a.beam, a.k))[: a.k]:
            print(r if a.k == 1 else f"{s:7.3f}  {r}")

    if a.text:
        run(" ".join(a.text))
        return
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("› "), None):
        if line.strip():
            run(line.strip())


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
