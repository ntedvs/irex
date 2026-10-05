"""Regex pool: sample in-scope JS regexes, synthesize positive/negative strings.

Pipeline per regex: string -> parse (scope check) -> sample positives -> mutate
into negatives -> label everything with real JS semantics (QuickJS).

    python src/grammar.py --n 60000
"""
import argparse
import json
import random
import string
import time
from pathlib import Path

import engine

# ---------------------------------------------------------------- char sets
U = frozenset(chr(c) for c in range(32, 127))  # printable ASCII universe
DIG = frozenset(string.digits)
WRD = frozenset(string.ascii_letters + string.digits + "_")
SPC = frozenset(" ")
ESC = {"d": DIG, "D": U - DIG, "w": WRD, "W": U - WRD, "s": SPC, "S": U - SPC}
SPECIAL = set("\\^$.|?*+()[]{}/")


# ---------------------------------------------------------------- parser
# AST: ("set", chars) | ("seq", [n]) | ("alt", [n]) | ("rep", n, lo, hi) | ("nil",)
class Parser:
    def __init__(self, s):
        self.s, self.i = s, 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def eat(self, c=None):
        ch = self.peek()
        if ch is None or (c and ch != c):
            raise ValueError(f"expected {c!r} at {self.i} in {self.s!r}")
        self.i += 1
        return ch

    def parse(self):
        n = self.alt()
        if self.peek() is not None:
            raise ValueError(f"trailing {self.s[self.i:]!r}")
        return n

    def alt(self):
        xs = [self.seq()]
        while self.peek() == "|":
            self.eat()
            xs.append(self.seq())
        return xs[0] if len(xs) == 1 else ("alt", xs)

    def seq(self):
        xs = []
        while self.peek() not in (None, "|", ")"):
            xs.append(self.quant(self.atom()))
        return ("seq", xs)

    def quant(self, a):
        c = self.peek()
        if c in ("*", "+", "?"):
            self.eat()
            lo, hi = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
        elif c == "{":
            self.eat()
            lo = self.num()
            hi = lo
            if self.peek() == ",":
                self.eat()
                hi = self.num() if self.peek() != "}" else None
            self.eat("}")
            if hi is not None and hi < lo:
                raise ValueError("bad range")
        else:
            return a
        if a[0] == "nil":
            raise ValueError("quantified anchor")
        if self.peek() == "?":  # lazy: same language for sampling
            self.eat()
        if self.peek() in ("*", "+", "?", "{"):
            raise ValueError("stacked quantifier")
        if hi is None and (nullable(a) or ambiguous(a)):
            raise ValueError("ambiguous unbounded repeat (ReDoS)")
        return ("rep", a, lo, hi)

    def num(self):
        j = self.i
        while (self.peek() or "").isdigit():
            self.i += 1
        if j == self.i:
            raise ValueError("expected number")
        return int(self.s[j:self.i])

    def atom(self):
        c = self.eat()
        if c == "(":
            if self.peek() == "?":
                self.eat()
                self.eat(":")  # only non-capturing allowed
            n = self.alt()
            self.eat(")")
            return n
        if c == "[":
            return self.cls()
        if c == ".":
            return ("set", U)
        if c in "^$":
            return ("nil",)
        if c == "\\":
            e = self.eat()
            if e in ESC:
                return ("set", ESC[e])
            if e in "bB":
                return ("nil",)
            if e in SPECIAL or e in "-":
                return ("set", frozenset(e))
            raise ValueError(f"out-of-scope escape \\{e}")
        if c in "*+?{}|)]":
            raise ValueError(f"unexpected {c!r}")
        return ("set", frozenset(c))

    def cls(self):
        neg = self.peek() == "^"
        if neg:
            self.eat()
        out, first = set(), True
        while self.peek() != "]" or first:
            first = False
            lo = self.cls_char()
            if isinstance(lo, frozenset):
                out |= lo
                continue
            if self.peek() == "-" and self.s[self.i + 1:self.i + 2] not in ("]", ""):
                self.eat()
                hi = self.cls_char()
                if isinstance(hi, frozenset) or ord(hi) < ord(lo):
                    raise ValueError("bad class range")
                out |= {chr(x) for x in range(ord(lo), ord(hi) + 1)}
            else:
                out.add(lo)
        self.eat("]")
        out &= U
        return ("set", frozenset(U - out if neg else out))

    def cls_char(self):
        c = self.eat()
        if c != "\\":
            return c
        e = self.eat()
        if e in ESC:
            return ESC[e]
        if e in SPECIAL or e in "-]^":
            return e
        raise ValueError(f"out-of-scope class escape \\{e}")


def nullable(n) -> bool:
    k = n[0]
    if k == "set":
        return False
    if k == "seq":
        return all(map(nullable, n[1]))
    if k == "alt":
        return any(map(nullable, n[1]))
    if k == "rep":
        return n[2] == 0 or nullable(n[1])
    return True


def _chars(n):
    """First-level char set of a quantified atom, else None."""
    while n[0] == "seq" and len(n[1]) == 1:
        n = n[1][0]
    if n[0] == "rep":
        return _chars(n[1])
    return n[1] if n[0] == "set" else None


def ambiguous(n) -> bool:
    r"""Body shapes that backtrack exponentially under an outer + or *:
    (a+)+, (\s*\s+)+, (\w+\d+)* ... i.e. a nested unbounded repeat, or two
    adjacent quantified atoms with overlapping chars."""
    while n[0] == "seq" and len(n[1]) == 1:
        n = n[1][0]
    if n[0] == "rep" and (n[3] is None or n[3] > 1):
        return True
    if n[0] == "alt":
        return any(map(ambiguous, n[1]))
    if n[0] != "seq":
        return False
    xs = n[1]
    for x, y in zip(xs, xs[1:]):
        if x[0] == "rep" and y[0] == "rep":
            cx, cy = _chars(x), _chars(y)
            if cx is None or cy is None or cx & cy:
                return True
    return False


def parse(pattern: str):
    return Parser(pattern).parse()


def in_scope(pattern: str, flags: str = "") -> bool:
    try:
        parse(pattern)
    except (ValueError, IndexError):
        return False
    return set(flags) <= {"i"} and engine.compiles(pattern, flags)


# ---------------------------------------------------------------- sampler
_SORTED: dict = {}


def _sorted(s):
    t = _SORTED.get(s)
    if t is None:
        t = _SORTED[s] = tuple(sorted(s))
    return t


def sample(n, rng: random.Random, ci=False) -> str:
    k = n[0]
    if k == "set":
        if not n[1]:
            raise ValueError("empty set")
        c = rng.choice(_sorted(n[1]))
        return c.swapcase() if ci and rng.random() < 0.3 else c
    if k == "seq":
        return "".join(sample(x, rng, ci) for x in n[1])
    if k == "alt":
        return sample(rng.choice(n[1]), rng, ci)
    if k == "rep":
        hi = n[3] if n[3] is not None else n[2] + rng.choice([0, 1, 2, 3, 5])
        return "".join(sample(n[1], rng, ci) for _ in range(rng.randint(n[2], min(hi, n[2] + 6))))
    return ""


NOISE = string.ascii_letters + string.digits + " .-_@/:#$,"


def mutate(s: str, rng: random.Random) -> str:
    op = rng.randrange(7)
    i = rng.randrange(len(s) + 1)
    c = rng.choice(NOISE)
    if op == 0 and s:
        return s[:i] + s[i + 1:]  # delete
    if op == 1:
        return s[:i] + c + s[i:]  # insert
    if op == 2 and s:
        i = min(i, len(s) - 1)
        return s[:i] + c + s[i + 1:]  # substitute
    if op == 3:
        return c + s  # prefix (breaks ^)
    if op == 4:
        return s + c  # suffix (breaks $)
    if op == 5 and len(s) > 1:
        return s[: rng.randrange(1, len(s))]  # truncate
    return s.swapcase()


def examples(pattern, flags, rng, k=5):
    """-> (pos, neg) labeled by JS, or None if degenerate or slow (ReDoS)."""
    t = time.perf_counter()
    try:
        ex = _examples(pattern, flags, rng, k)
    except Exception:  # QuickJS time limit, empty set, ...
        return None
    return ex if time.perf_counter() - t < 0.05 else None


def _examples(pattern, flags, rng, k):
    tree = parse(pattern)
    ci = "i" in flags
    cand = list(dict.fromkeys(sample(tree, rng, ci) for _ in range(24)))
    cand = [s for s in cand if s and len(s) <= 60]
    if len(cand) < 3:
        return None
    ok = engine.test(pattern, flags, cand)
    pos = [s for s, m in zip(cand, ok) if m]
    if len(pos) < 3:
        return None
    muts = [m for m in dict.fromkeys(mutate(rng.choice(pos), rng) for _ in range(40)) if m not in pos]
    muts += ["".join(rng.choices(NOISE, k=rng.randint(1, 12))) for _ in range(6)]
    ok = engine.test(pattern, flags, muts)
    neg = [s for s, m in zip(muts, ok) if not m and s.strip()]
    if len(neg) < 3:
        return None
    rng.shuffle(pos), rng.shuffle(neg)
    return pos[:k], neg[:k]


# ---------------------------------------------------------------- generator
WORDS = """cat dog bird fish red green blue black white gray error warn info debug
fatal http https ftp www com org net io dev app api user admin root guest test
id key token name file data log img src lib bin tmp home yes no true false null
on off get post put delete open close start stop begin end jan feb mar apr may
jun jul aug sep oct nov dec mon tue wed thu fri sat sun am pm usd eur gbp jpg
png gif pdf txt csv json xml html css js py md zip tar gz mp3 mp4 wav todo fixme
note hello world foo bar baz qux the and or not color colour gray grey cancel
ok fail pass high low new old big small north south east west""".split()

CLASSES = ["[a-z]", "[A-Z]", "[a-zA-Z]", "[0-9]", "[a-z0-9]", "[A-Za-z0-9_]",
           "[aeiou]", "[^aeiou]", "[a-f0-9]", "[A-F0-9]", "[0-7]", "[1-9]",
           "[01]", "[^0-9]", "[^\\s]", "[-_.]", "[.,;:!?]", "[+-]", "[a-zA-Z ]",
           "[^a-z]", "[A-Z0-9]", "[a-zA-Z0-9.-]", "[^,]", "[^\"]", "[xyz]"]
ESCAPES = ["\\d", "\\w", "\\s", "\\D", "\\W", "\\S", "."]
SEPS = ["-", "\\.", "_", ":", "\\/", " ", "@", ",", "\\s", "\\s*", "\\s+", "#", "="]


def lit(s):
    return "".join("\\" + c if c in SPECIAL else c for c in s)


def rand_class(rng):
    if rng.random() < 0.2:
        cs = "".join(sorted(rng.sample(string.ascii_lowercase, rng.randint(2, 5))))
        return f"[{'^' if rng.random() < 0.2 else ''}{cs}]"
    return rng.choice(CLASSES)


def quant(rng, allow_none=True):
    r = rng.random()
    lo = rng.randint(1, 5)
    q = ("" if allow_none else "+") if r < 0.25 else \
        "?" if r < 0.35 else "*" if r < 0.47 else "+" if r < 0.65 else \
        f"{{{lo}}}" if r < 0.8 else f"{{{lo},}}" if r < 0.86 else f"{{{lo},{lo + rng.randint(1, 6)}}}"
    return q + ("?" if q and rng.random() < 0.04 else "")


def atom(rng):
    r = rng.random()
    if r < 0.5:
        return rand_class(rng)
    if r < 0.85:
        return rng.choice(ESCAPES)
    return lit(rng.choice(string.ascii_lowercase + "-_.@#$%!"))


def group(rng, body):
    return f"({'?:' if rng.random() < 0.4 else ''}{body})"


def chunk(rng, depth=0):
    r = rng.random()
    if r < 0.18:
        return lit(rng.choice(WORDS))
    if r < 0.55:
        return atom(rng) + quant(rng)
    if r < 0.72:
        alts = rng.sample(WORDS, rng.randint(2, 4))
        return group(rng, "|".join(map(lit, alts))) + ("?" if rng.random() < 0.15 else "")
    if r < 0.85 and depth < 1:
        body = rng.choice(SEPS) + chunk(rng, depth + 1)
        return group(rng, body) + rng.choice(["?", "*", "+", "?", f"{{{rng.randint(2, 4)}}}"])
    return rng.choice(SEPS)


def anchor(rng, body):
    r = rng.random()
    if r < 0.4:
        return f"^{body}$"
    if r < 0.55:
        return f"^{body}"
    if r < 0.65:
        return f"{body}$"
    if r < 0.8:
        return f"\\b{body}\\b"
    return body


def random_regex(rng):
    body = "".join(chunk(rng) for _ in range(rng.choice([1, 1, 2, 2, 2, 3, 3, 4])))
    p = anchor(rng, body)
    letters = any(c.isalpha() for c in p.replace("\\", " \\"))
    return p, ("i" if letters and rng.random() < 0.15 else "")


def C(*xs):
    return lambda rng: rng.choice(xs)


def opt(rng, s, p=0.5):
    return s if rng.random() < p else ""


# realistic shapes; each call randomizes details
TEMPLATES = [
    lambda r: f"^\\d{{{r.choice([3, 4, 5, 6, 8])}}}$",
    lambda r: "^\\d{5}" + opt(r, "(?:-\\d{4})?") + "$",
    lambda r: "^" + opt(r, "\\(?") + "\\d{3}" + opt(r, "\\)?") + C("-", "[-.\\s]?", "\\s?", "[-. ]")(r) + "\\d{3}" + C("-", "[-.]?", "[-. ]")(r) + "\\d{4}$",
    lambda r: (lambda s: f"^\\d{{4}}{s}\\d{{2}}{s}\\d{{2}}$")(C("-", "\\/", "\\.")(r)),
    lambda r: (lambda s: f"^\\d{{1,2}}{s}\\d{{1,2}}{s}\\d{{{r.choice(['2', '4', '2,4'])}}}$")(C("\\/", "-", "\\.")(r)),
    lambda r: "^" + C("\\d{2}", "[01]\\d", "\\d{1,2}", "(?:[01]\\d|2[0-3])")(r) + ":[0-5]\\d" + opt(r, "(?::[0-5]\\d)?") + opt(r, "\\s?(?:am|pm)", 0.3) + "$",
    lambda r: "^#" + C("[0-9a-fA-F]{6}", "[0-9a-f]{6}", "(?:[0-9a-fA-F]{3}){1,2}", "[A-F0-9]{6}")(r) + "$",
    lambda r: "^" + C("\\d{1,3}(?:\\.\\d{1,3}){3}", "(?:\\d{1,3}\\.){3}\\d{1,3}")(r) + "$",
    lambda r: "^" + C("[\\w.+-]+", "[a-z0-9._%+-]+", "\\w+", "[a-zA-Z0-9._-]+")(r) + "@" + C("[\\w-]+", "[a-z0-9-]+", "\\w+")(r) + C("\\.[a-z]{2,}", "\\.[a-zA-Z]{2,6}", "(?:\\.\\w+)+", "\\.(?:com|org|net)")(r) + "$",
    lambda r: "^" + C("https?", "https", "(?:https?|ftp)")(r) + ":\\/\\/" + opt(r, "(?:www\\.)?") + C("[\\w-]+(?:\\.[\\w-]+)+", "[^\\s\\/]+", "[a-z0-9.-]+\\.[a-z]{2,}")(r) + opt(r, "(?:\\/\\S*)?") + "$",
    lambda r: "\\." + group(r, "|".join(r.sample(["jpg", "jpeg", "png", "gif", "webp", "svg", "bmp"], r.randint(2, 4)))) + "$",
    lambda r: "\\." + group(r, "|".join(r.sample(["pdf", "docx?", "txt", "md", "csv", "xlsx?", "json", "ya?ml"], r.randint(2, 4)))) + "$",
    lambda r: C("#\\w+", "#[a-zA-Z]\\w*", "\\B#\\w+", "#[a-z0-9_]+")(r),
    lambda r: C("@\\w+", "@[a-zA-Z0-9_]{1,15}", "\\B@\\w+")(r),
    lambda r: "^\\$" + C("\\d+", "\\d{1,3}(?:,\\d{3})*")(r) + opt(r, "(?:\\.\\d{2})?", 0.7) + "$",
    lambda r: "^[a-z" + opt(r, "A-Z") + "][a-z" + opt(r, "A-Z") + "0-9_" + opt(r, "-") + f"]{{{r.randint(2, 4)},{r.randint(12, 20)}}}$",
    lambda r: "^" + opt(r, "v?") + "\\d+\\.\\d+" + C("\\.\\d+", "(?:\\.\\d+)?")(r) + opt(r, "(?:-[a-z0-9.]+)?", 0.3) + "$",
    lambda r: "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    lambda r: "^" + C("\\d{4}(?:[- ]?\\d{4}){3}", "(?:\\d{4}[- ]?){3}\\d{4}", "\\d{16}")(r) + "$",
    lambda r: "^" + opt(r, "-?") + C("\\d+", "\\d{1,3}", "100|\\d{1,2}", "\\d+(?:\\.\\d+)?")(r) + "%$",
    lambda r: "^" + opt(r, "[+-]?") + C("\\d+", "\\d*\\.\\d+", "\\d+(?:\\.\\d+)?", "\\d+(?:\\.\\d*)?")(r) + opt(r, "(?:[eE][+-]?\\d+)?", 0.25) + "$",
    lambda r: "^0x[0-9a-fA-F]+$" if r.random() < 0.5 else "^(?:0x)?[0-9a-f]+$",
    lambda r: "^[01]+$" if r.random() < 0.5 else f"^[01]{{{r.choice([4, 8, 16])}}}$",
    lambda r: f"\\b{lit(r.choice(WORDS))}\\w*" if r.random() < 0.5 else f"\\w*{lit(r.choice(['ing', 'ed', 'ly', 'tion', 's', 'er']))}\\b",
    lambda r: f"^{lit(r.choice(WORDS))}" + C("\\b", ".*", "\\s", ":")(r),
    lambda r: f"{lit(r.choice(WORDS))}$",
    lambda r: "\\b" + group(r, "|".join(map(lit, r.sample(WORDS, r.randint(2, 5))))) + "\\b",
    lambda r: "^" + group(r, "|".join(map(lit, r.sample(WORDS, r.randint(2, 5))))) + "$",
    lambda r: "^" + C("[A-Z]", "[A-Z][a-z]+", "[a-z]")(r) + C("[a-z]*", "\\w*", "[a-z]+(?: [a-z]+)*")(r) + opt(r, "[.!?]") + "$",
    lambda r: "^\\s*$" if r.random() < 0.3 else C("^\\s+", "\\s+$", "\\s{2,}", "^\\s+|\\s+$")(r),
    lambda r: "^" + C("[A-Z]{2}", "[A-Z]{3}", "[A-Z]{2,3}")(r) + C("-", "", "\\s?", "_")(r) + f"\\d{{{r.randint(2, 6)}}}$",
    lambda r: "^[A-Z]{1,2}\\d[A-Z\\d]? ?\\d[A-Z]{2}$",
    lambda r: "<" + C("[a-z]+", "\\/?[a-z]+", "[a-z][a-z0-9]*")(r) + C(">", "[^>]*>", "\\s*\\/?>")(r),
    lambda r: "\"" + C("[^\"]*", "[^\"]+", ".*?")(r) + "\"",
    lambda r: "\\(" + C("[^)]*", "\\d+", "\\w+")(r) + "\\)",
    lambda r: "\\[" + C("[^\\]]*", "\\d+", "\\w+")(r) + "\\]",
    lambda r: "^" + group(r, "|".join(r.sample(["ERROR", "WARN", "INFO", "DEBUG", "FATAL", "TRACE"], r.randint(2, 4)))) + C(":", "\\s", "\\b", ":\\s.*")(r),
    lambda r: "^\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}" + opt(r, ":\\d{2}") + opt(r, "(?:Z|[+-]\\d{2}:\\d{2})?", 0.3) + "$",
    lambda r: "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$",
    lambda r: "^" + C("\\d+", "[1-9]\\d*", "[1-9]\\d{0,2}", "0|[1-9]\\d*")(r) + "$",
    lambda r: "^(?:" + C("\\w+", "[a-z]+", "\\d+")(r) + C(",", ";", "\\|", ",\\s*")(r) + ")*" + C("\\w+", "[a-z]+", "\\d+")(r) + "$",
    lambda r: "^" + C("[A-Za-z]{2,}", "[A-Z][a-z]+")(r) + " " + C("[A-Za-z]{2,}", "[A-Z][a-z]+")(r) + "$",
    lambda r: C("colou?r", "gr[ae]y", "(?:cancell?ed)", "favou?rite", "analy[sz]e")(r),
    lambda r: C("\\d+(?:px|em|rem|%)", "-?\\d+(?:\\.\\d+)?(?:px|em|rem)", "\\d+(?:ms|s)")(r),
    lambda r: "^[a-z]+(?:-[a-z]+)*$" if r.random() < 0.5 else C("^[a-z]+(?:_[a-z]+)*$", "^[a-z]+(?:[A-Z][a-z]+)*$", "^[A-Z][a-z]+(?:[A-Z][a-z]+)*$")(r),
    lambda r: "^\\/" + C("[\\w-]+(?:\\/[\\w-]+)*", "(?:[\\w.-]+\\/)*[\\w.-]+", "[a-z]+\\/\\d+")(r) + opt(r, "\\/?") + "$",
    lambda r: "^" + C("[A-Z]{3}", "[A-Z]{2}")(r) + "\\s?\\d+(?:\\.\\d{2})?$",
    lambda r: "\\b\\d{1,2}(?:st|nd|rd|th)\\b",
    lambda r: "^(?:" + "|".join(r.sample(["mon", "tue", "wed", "thu", "fri", "sat", "sun"], r.randint(3, 7))) + ")$",
    lambda r: "^(?:" + "|".join(r.sample(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], r.randint(3, 6))) + ")" + opt(r, "[a-z]*") + "$",
    lambda r: "^[a-zA-Z0-9]{" + str(r.choice([6, 8, 10, 12])) + ",}$",
    lambda r: "^[^\\s]+$" if r.random() < 0.5 else "^\\S+$",
    lambda r: "^.{" + str(r.randint(1, 5)) + "," + str(r.randint(8, 40)) + "}$",
]


def flags_for(rng, p):
    return "i" if any(c.isalpha() for c in p if c not in "dwsDWSbB") and rng.random() < 0.15 else ""


def regex(rng):
    if rng.random() < 0.5:
        p = rng.choice(TEMPLATES)(rng)
        return p, flags_for(rng, p)
    return random_regex(rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--out", default=Path(__file__).parents[1] / "data/regex.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng, seen, rows, tries = random.Random(a.seed), set(), 0, 0
    with open(a.out, "w") as f:
        while rows < a.n and tries < a.n * 20:
            tries += 1
            p, fl = regex(rng)
            key = (p, fl)
            if key in seen or not (2 <= len(p) <= 72) or not in_scope(p, fl):
                continue
            seen.add(key)
            ex = examples(p, fl, rng)
            if not ex:
                continue
            f.write(json.dumps({"id": rows, "regex": engine.literal(p, fl), "pattern": p,
                                "flags": fl, "pos": ex[0], "neg": ex[1]}) + "\n")
            rows += 1
    print(f"{rows} regexes from {tries} tries -> {a.out}")


if __name__ == "__main__":
    main()
