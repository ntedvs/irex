"""Metrics: exact match, compile rate, behavioral equivalence.

Behavioral equivalence: gold and predicted regex agree on gold's pos/neg
strings plus strings sampled from (and mutated off) the prediction.

    python src/evaluate.py --ckpt ckpt/best --split test
"""
import argparse
import json
import random
from pathlib import Path

import engine
import grammar

ROOT = Path(__file__).parents[1]


def equivalent(pred: str, gold: str, pos, neg, rng=None) -> bool:
    rng = rng or random.Random(0)
    try:
        pp, pf = engine.parse_literal(pred)
        gp, gf = engine.parse_literal(gold)
        if not engine.compiles(pp, pf):
            return False
        probe = list(pos) + list(neg)
        try:
            tree = grammar.parse(pp)
            s = [grammar.sample(tree, rng, "i" in pf) for _ in range(8)]
            probe += s + [grammar.mutate(x, rng) for x in s]
        except Exception:
            pass
        return engine.test(pp, pf, probe) == engine.test(gp, gf, probe)
    except Exception:
        return False


def score(preds, rows):
    n = len(rows)
    ex = sum(p == r["re"] for p, r in zip(preds, rows))
    comp = 0
    for p in preds:
        try:
            comp += engine.compiles(*engine.parse_literal(p))
        except Exception:
            pass
    eq = sum(p == r["re"] or equivalent(p, r["re"], r["pos"], r["neg"]) for p, r in zip(preds, rows))
    return {"exact": ex / n, "equiv": eq / n, "compiles": comp / n, "n": n}


def by_style(preds, rows):
    out = {}
    for s in sorted({r["style"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["style"] == s]
        out[s] = score([preds[i] for i in idx], [rows[i] for i in idx])["equiv"]
    return out


def main():
    import infer
    import tokenizer as tok
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ROOT / "ckpt/best")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--beam", type=int, default=0, help="0 = greedy")
    ap.add_argument("--bits", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT / f"data/{a.split}.jsonl")][: a.n]
    model = infer.load(a.ckpt, a.bits)
    if a.beam:
        preds = [next(iter(infer.best(model, r["en"], k=a.beam)), ("",))[0] for r in rows]
    else:
        preds = []
        for i in range(0, len(rows), 256):
            preds += infer.greedy(model, [tok.prompt(r["en"]) for r in rows[i:i + 256]])
    print(json.dumps(score(preds, rows)))
    with open(Path(a.ckpt) / f"preds_{a.split}{'_q%d' % a.bits if a.bits else ''}.jsonl", "w") as f:
        for p, r in zip(preds, rows):
            f.write(json.dumps({"en": r["en"], "gold": r["re"], "pred": p, "style": r["style"],
                                "ok": p == r["re"] or equivalent(p, r["re"], r["pos"], r["neg"])}) + "\n")
    names = ["casual", "terse", "precise", "purpose", "sloppy", "short"]
    print({names[k]: round(v, 3) for k, v in by_style(preds, rows).items()})


if __name__ == "__main__":
    main()
