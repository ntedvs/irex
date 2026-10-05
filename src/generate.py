"""English descriptions for regexes via DeepSeek (deepseek-flash, non-thinking).

Resumable: skips regex ids already in the output. Stops at --budget dollars,
priced conservatively at peak rates.

    python src/generate.py --limit 40          # smoke test, ~cents
    python src/generate.py --budget 4.5        # the real run
"""
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).parents[1]
PRICE = {"peak": (0.006, 0.30, 1.20), "offpeak": (0.003, 0.15, 0.60)}  # $/M: cache hit, miss, out

SYSTEM = r"""You write the English requests people type when they want a specific JavaScript regex.

For EACH regex you get, write 5 requests, one per style, in this order:
1. casual: plain everyday words, as a non-programmer would ask a friend
2. terse: shortest complete wording, fragments fine, like a commit message
3. precise: careful spec in full sentences, every detail spelled out
4. purpose: lead with what it is for if it has a recognizable use (zip code, date, file extension...), otherwise phrase it as a task ("find...", "check that...", "grab..."); still include every detail needed to rebuild the exact regex
5. sloppy: all lowercase, rushed, minimal punctuation, maybe a typo

Rules:
- WORDS ONLY. Never write regex syntax or symbols-as-syntax: no backslashes, brackets, braces, pipes, ^, $, *, +, ?, "(?:". Literal characters the text must contain are fine, quoted: a "-", the "@" sign, a ".".
- Each request must pin down the behavior: which characters, exact counts and ranges, optional parts, alternatives, literal text.
- Convey anchoring in natural words: whole string / entire input / exactly; starts with; ends with; as a whole word; anywhere / contains.
- If the flag is i, say case doesn't matter (any phrasing).
- Vary vocabulary and sentence shape; avoid repeating openers like "Starts with" or "Matches" across the 5. Never begin with "Regex".
- One line each, 3 to 40 words.
- The examples are only hints; describe the pattern, not the examples.

Example input:
0. /^\d{5}(?:-\d{4})?$/    e.g. ["90210", "12345-6789"]
Example output:
{"0": ["a US zip code, five numbers and maybe a dash and four more", "5-digit zip, optional dash + 4 digits, nothing else", "The entire input must be exactly five digits, optionally followed by a hyphen and exactly four digits.", "validate ZIP or ZIP+4 codes", "zip code 5 digits optionaly dash 4 more"]}

Reply with JSON only, keyed by input index: {"0": [5 strings], "1": [5 strings], ...}"""


SHORT = r"""You write the SHORT requests people actually type into a "describe it, get a regex" box.

For EACH JavaScript regex you get, write 8 different short requests, 2 to 12 words each:
- name the thing when it has a common name (zip code, hex color, ISO date, email, US phone, file extension, hashtag...)
- still mention any detail that makes this regex differ from the obvious version (optional parts, separators, counts, whole string vs anywhere, case-insensitive)
- mix: noun phrases ("us zip code"), commands ("match a hex color"), questions ("how do i check for a date like 2024-01-31"), lowercase sloppy, and ones with an example value
- words only: no regex syntax, no backslashes or brackets

Example input:
0. /^\d{5}(?:-\d{4})?$/    e.g. ["90210", "12345-6789"]
Example output:
{"0": ["us zip code with optional +4", "zip or zip+4", "validate a ZIP code like 90210 or 90210-1234", "5 digit zip, optional dash and 4 digits", "match zip codes", "zip code, entire string", "is this a valid us postal code", "zipcode 5 digits maybe -4 more"]}

Reply with JSON only, keyed by input index: {"0": [8 strings], ...}"""


def prompt(batch):
    return "\n".join(f"{i}. {r['regex']}    e.g. {json.dumps(r['pos'][:2])}" for i, r in enumerate(batch))


class Meter:
    def __init__(self):
        self.hit = self.miss = self.out = self.pairs = self.fails = 0

    def add(self, u):
        hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        self.hit += hit
        self.miss += u.prompt_tokens - hit
        self.out += u.completion_tokens

    def cost(self, tier="peak"):
        h, m, o = PRICE[tier]
        return (self.hit * h + self.miss * m + self.out * o) / 1e6

    def __str__(self):
        return (f"pairs={self.pairs} fails={self.fails} tok(hit/miss/out)={self.hit}/{self.miss}/{self.out} "
                f"${self.cost('peak'):.4f} peak / ${self.cost('offpeak'):.4f} off-peak")


async def describe(client, batch, meter, system=SYSTEM, retries=3):
    for attempt in range(retries):
        try:
            r = await client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt(batch)}],
                response_format={"type": "json_object"},
                max_tokens=6000,
                temperature=1.0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            meter.add(r.usage)
            data = json.loads(r.choices[0].message.content)
            out = []
            for i, row in enumerate(batch):
                ds = data.get(str(i))
                if isinstance(ds, list):
                    ds = [d.strip() for d in ds if isinstance(d, str) and d.strip()]
                    out.append({"id": row["id"], "regex": row["regex"], "en": ds})
            return out
        except Exception as e:  # rate limit, bad JSON, network
            if attempt == retries - 1:
                print(f"  batch {batch[0]['id']} failed: {type(e).__name__}: {e}")
                meter.fails += 1
                return []
            await asyncio.sleep(2 ** attempt + random.random())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=ROOT / "data/regex.jsonl")
    ap.add_argument("--out", default=ROOT / "data/raw.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="max regexes this run")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--budget", type=float, default=0.10, help="stop at this many $ (peak pricing)")
    ap.add_argument("--short", action="store_true", help="short real-world requests (SHORT prompt)")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [r for r in map(json.loads, open(a.src)) if r["id"] not in done][: a.limit]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{len(done)} done, {len(todo)} to go in {len(batches)} batches")

    meter, sem, t0 = Meter(), asyncio.Semaphore(a.concurrency), time.time()
    f = open(a.out, "a")

    async def run(b):
        async with sem:
            if meter.cost() >= a.budget:
                return
            for row in await describe(client, b, meter, SHORT if a.short else SYSTEM):
                row["short"] = a.short
                f.write(json.dumps(row) + "\n")
                meter.pairs += len(row["en"])
            f.flush()

    tasks = [asyncio.create_task(run(b)) for b in batches]
    pending = set(tasks)
    while pending:
        _, pending = await asyncio.wait(pending, timeout=15)
        print(f"[{time.time() - t0:5.0f}s] {meter}", flush=True)
    f.close()
    print("done:", meter)


if __name__ == "__main__":
    asyncio.run(main())
