"""Clean + split generated pairs.

raw.jsonl (id, regex, en[]) + regex.jsonl (pos, neg) -> train/val/test.jsonl
Split is by regex id, so test regexes are never seen in training.

    python src/filter.py
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
LEAK = re.compile(r"[\\\[\]{}|]|\(\?:|\.\*|\\[dws]")


def bucket(regex: str) -> str:
    """Split by regex text, so the same regex never lands in two splits."""
    h = int(hashlib.md5(regex.encode()).hexdigest(), 16) % 100
    return "test" if h < 2 else "val" if h < 4 else "train"


def clean(d: str, regex: str, pattern: str) -> str | None:
    d = " ".join(d.split())
    if not (3 <= len(d) <= 300) or LEAK.search(d) or pattern in d or regex in d:
        return None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=ROOT / "data/raw.jsonl")
    ap.add_argument("--pool", default=ROOT / "data/regex.jsonl")
    a = ap.parse_args()

    pools = [a.pool] + ([ROOT / "data/regex_short.jsonl"] if (ROOT / "data/regex_short.jsonl").exists() else [])
    pool = {r["id"]: r for path in pools for r in map(json.loads, open(path))}
    out = {k: open(ROOT / f"data/{k}.jsonl", "w") for k in ("train", "val", "test")}
    n = {k: 0 for k in out}
    seen, kept, dropped = set(), 0, 0
    raws = [a.raw] + ([ROOT / "data/short.jsonl"] if (ROOT / "data/short.jsonl").exists() else [])
    for row in (json.loads(l) for path in raws for l in open(path)):
        if row["id"] in seen or row["id"] not in pool:
            continue
        seen.add(row["id"])
        p = pool[row["id"]]
        k = bucket(p["regex"])
        for style, d in enumerate(row["en"][:8]):
            style = 5 if row.get("short") else style
            d = clean(d, p["regex"], p["pattern"])
            if d is None:
                dropped += 1
                continue
            kept += 1
            rec = {"id": p["id"], "en": d, "re": p["regex"], "style": style}
            if k != "train":
                rec |= {"pos": p["pos"], "neg": p["neg"]}
            out[k].write(json.dumps(rec) + "\n")
            n[k] += 1
    common = ROOT / "data/common.jsonl"
    if common.exists():  # rule-based everyday pairs: train only (benchmark is the real test)
        for l in open(common):
            out["train"].write(l)
            n["train"] += 1
    print(f"regexes={len(seen)} kept={kept} dropped={dropped} {n}")


if __name__ == "__main__":
    main()
