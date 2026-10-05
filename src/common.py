"""Rule-based English for everyday regexes (free, no LLM).

Covers common intents the LLM data under-represents: char-class-only strings,
lengths, passwords, starts/ends/contains, find-in-text, decimals, uuids...

    python src/common.py --n 60000      # -> data/common.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import engine

ROOT = Path(__file__).parents[1]
R = random.Random()


def pick(*xs):
    return R.choice(xs)


def maybe(s, p=0.5):
    return s if R.random() < p else ""


SPECIAL = set("\\^$.|?*+()[]{}/")


def lit(s):
    return "".join("\\" + c if c in SPECIAL else c for c in s)


# name variants -> class
CLASSES = [
    (["digits", "numbers", "numeric characters", "digits 0-9"], "\\d", "[0-9]"),
    (["letters", "alphabetic characters", "letters a-z"], "[a-zA-Z]", "[A-Za-z]"),
    (["lowercase letters", "lower case letters", "small letters"], "[a-z]", None),
    (["uppercase letters", "capital letters", "upper case letters", "caps"], "[A-Z]", None),
    (["letters and numbers", "letters or digits", "alphanumeric characters", "alphanumerics"], "[a-zA-Z0-9]", "[A-Za-z0-9]"),
    (["word characters", "letters, digits or underscores", "letters numbers and underscores"], "\\w", "[a-zA-Z0-9_]"),
    (["hex digits", "hexadecimal characters"], "[0-9a-fA-F]", "[0-9a-f]"),
    (["vowels"], "[aeiou]", None),
    (["0s and 1s", "ones and zeros", "binary digits"], "[01]", None),
]
CHARS = {";": ["a semicolon", "semicolon", "a \";\""], ".": ["a period", "a dot", "a full stop"],
         ",": ["a comma", "comma"], ":": ["a colon", "colon"], "?": ["a question mark", "question mark"],
         "!": ["an exclamation mark", "an exclamation point", "a bang"], "/": ["a slash", "a forward slash"],
         "-": ["a dash", "a hyphen"], "_": ["an underscore"], "@": ["an at sign", "an \"@\""],
         "#": ["a hash", "a pound sign", "a \"#\""], "$": ["a dollar sign"], "%": ["a percent sign"],
         ")": ["a closing parenthesis"], "\"": ["a double quote"], "'": ["a single quote", "an apostrophe"]}
WORDS = ("error warning todo fixme note info debug fatal hello world foo bar test admin user id name "
         "import from def class return http www api key token true false null yes no ok done "
         "cat dog red blue green apple banana color size price total date time").split()
UPPER = "TODO FIXME NOTE ERROR WARN INFO DEBUG BUG HACK XXX".split()


def only():
    names, c, alt = R.choice(CLASSES)
    n = R.choice(names)
    rx = f"^{R.choice([c, alt]) if alt else c}+$"
    en = pick(f"only {n}", f"{n} only", f"nothing but {n}", f"string made of only {n}",
              f"must contain only {n}", f"all {n}", f"validate that it's just {n}", f"only {n}, nothing else",
              f"check the whole thing is {n}", f"one or more {n} and nothing else")
    return rx, en


def length():
    names, c, alt = R.choice(CLASSES + [(["characters", "chars", "of any character"], ".", None)])
    n = R.choice(names).replace("of any character", "characters")
    c = R.choice([c, alt]) if alt else c
    a = R.randint(1, 12)
    b = a + R.randint(1, 12)
    k = R.randrange(3)
    if k == 0:
        rx = f"^{c}{{{a}}}$"
        en = pick(f"exactly {a} {n}", f"{a} {n}", f"exactly {a} {n}, nothing else", f"{a} {n} long",
                  f"a string of exactly {a} {n}", f"must be {a} {n}")
    elif k == 1:
        rx = f"^{c}{{{a},}}$"
        en = pick(f"at least {a} {n}", f"{a} or more {n}", f"minimum {a} {n}", f"no fewer than {a} {n}",
                  f"{a}+ {n}")
    else:
        rx = f"^{c}{{{a},{b}}}$"
        en = pick(f"{a} to {b} {n}", f"between {a} and {b} {n}", f"{a}-{b} {n}", f"from {a} up to {b} {n}",
                  f"{n}, {a} to {b} long")
    return rx, en


def password():
    a = R.randint(6, 16)
    names, c, alt = R.choice(CLASSES[1:6])
    n = R.choice(names)
    c = R.choice([c, alt]) if alt else c
    if R.random() < 0.5:
        rx = f"^{c}{{{a},}}$"
        en = pick(f"password at least {a} characters, only {n}", f"{n} only password, min {a} chars",
                  f"password of {a} or more {n}", f"a password with at least {a} {n}",
                  f"pw at least {a} long, {n} only")
    else:
        b = a + R.randint(4, 24)
        rx = f"^{c}{{{a},{b}}}$"
        en = pick(f"password {a} to {b} characters, {n} only", f"password between {a} and {b} {n}",
                  f"{a}-{b} character password using {n}")
    return rx, en


def starts_ends():
    ch = R.choice(list(CHARS))
    name = R.choice(CHARS[ch])
    w = R.choice(WORDS)
    k = R.randrange(6)
    if k == 0:
        return f"{lit(ch)}$", pick(f"ends with {name}", f"lines that end with {name}", f"ending in {name}",
                                    f"strings ending with {name}", f"last character is {name}")
    if k == 1:
        return f"^{lit(ch)}", pick(f"starts with {name}", f"lines beginning with {name}", f"first character is {name}",
                                    f"begins with {name}")
    if k == 2:
        return f"^{lit(w)}", pick(f"starts with {w}", f"lines starting with {w}", f"begins with the word {w}",
                                   f"text that starts with \"{w}\"")
    if k == 3:
        return f"{lit(w)}$", pick(f"ends with {w}", f"lines ending in {w}", f"string ends with \"{w}\"")
    if k == 4:
        return f"\\b{lit(w)}\\b", pick(f"contains the word {w}", f"find the word {w}", f"the whole word {w}",
                                       f"{w} as a whole word", f"match the word \"{w}\"")
    return lit(ch), pick(f"contains {name}", f"has {name} anywhere", f"any {name}")


def upper_tag():
    t = R.choice(UPPER)
    return f"^{t}", pick(f"lines starting with {t}", f"starts with {t}", f"{t} at the start of the line",
                          f"lines that begin with {t}", f"find {t} lines")


def find():
    k = R.randrange(6)
    if k == 0:
        return "\\d+", pick("find all numbers in a string", "numbers anywhere in the text", "find digits",
                            "grab every number", "any sequence of digits", "extract numbers")
    if k == 1:
        return "\\w+", pick("find all words", "match each word", "grab words", "every word in the text")
    if k == 2:
        return "\\b[A-Z][a-z]*\\b", pick("find capitalized words", "words starting with a capital letter",
                                         "grab words that begin uppercase")
    if k == 3:
        n = R.randint(2, 8)
        return f"\\b\\d{{{n}}}\\b", pick(f"find {n} digit numbers", f"any {n}-digit number in the text",
                                         f"grab numbers with exactly {n} digits")
    if k == 4:
        return "\\s{2,}", pick("two or more spaces in a row", "multiple whitespace", "find double spaces",
                               "runs of whitespace")
    return "[A-Z]{2,}", pick("find all caps words", "sequences of uppercase letters", "acronyms in the text")


def capital():
    k = R.randrange(4)
    if k == 0:
        return "^[A-Z]", pick("starts with a capital letter", "first letter uppercase", "begins with an uppercase letter")
    if k == 1:
        return "^[a-z]", pick("starts with a lowercase letter", "begins lowercase", "first char is a small letter")
    if k == 2:
        return "^\\d", pick("starts with a digit", "begins with a number", "first character is a number")
    return "\\d$", pick("ends with a digit", "ends in a number", "last character is a digit")


def number():
    k = R.randrange(8)
    opts = [
        ("^\\d+\\.\\d+$", ["a decimal number like 3.14", "decimal number", "number with a decimal point",
                           "digits dot digits", "a float like 2.5"]),
        ("^-?\\d+$", ["an integer, optionally negative", "whole number with optional minus", "signed integer"]),
        ("^-?\\d+(?:\\.\\d+)?$", ["a number with optional decimals", "integer or decimal, maybe negative",
                                  "any number like -3 or 4.25"]),
        ("^[1-9]\\d*$", ["positive integer, no leading zeros", "a positive whole number"]),
        ("^\\d+(?:\\.\\d{1,2})?$", ["price with up to 2 decimals", "amount like 10 or 10.5 or 10.99"]),
        ("^\\d+\\.\\d+\\.\\d+$", ["semantic version like 1.2.3", "version number x.y.z", "semver",
                                   "three numbers separated by dots"]),
        ("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
         ["a uuid", "uuid like 123e4567-e89b-12d3-a456-426614174000", "guid", "match a UUID"]),
        ("^[A-Z]{2}$", ["us state abbreviation, two capital letters", "two letter state code", "2 uppercase letters"]),
    ]
    rx, ens = opts[k]
    return rx, R.choice(ens)


def codes():
    k = R.randrange(5)
    if k == 0:
        return "^\\d{5}(?:-\\d{4})?$", pick("us zip code with optional +4", "zip or zip+4", "zip code, optional dash and 4 digits",
                                            "5 digit zip with optional extension")
    if k == 1:
        return "^#(?:[0-9a-fA-F]{3}){1,2}$", pick("hex color like #fff or #a1b2c3", "css hex color, 3 or 6 digits",
                                                   "hex color code", "a color like #FF00AA or #abc")
    if k == 2:
        return "^#[0-9a-fA-F]{6}$", pick("6 digit hex color", "hex color like #ff00aa", "#rrggbb color")
    if k == 3:
        return "^\\s*$", pick("empty or whitespace-only line", "blank line", "nothing but spaces")
    return "^$", pick("empty string", "an empty line", "nothing at all")


def oneof():
    ws = R.sample(WORDS, R.randint(2, 4))
    rx = "^(?:" + "|".join(map(lit, ws)) + ")$"
    j = " or ".join(ws) if len(ws) == 2 else ", ".join(ws[:-1]) + " or " + ws[-1]
    return rx, pick(f"exactly {j}", f"either {j}", f"one of {j}", f"only the words {j}", f"{j}, nothing else")


GEN = [(only, 3), (length, 3), (password, 2), (starts_ends, 4), (upper_tag, 1), (find, 2),
       (capital, 1), (number, 2), (codes, 1), (oneof, 1)]


def sample():
    f = R.choices([g for g, _ in GEN], [w for _, w in GEN])[0]
    rx, en = f()
    flags = ""
    if R.random() < 0.08 and any(c.isalpha() for c in rx.replace("\\d", "").replace("\\w", "").replace("\\s", "").replace("\\b", "")):
        flags = "i"
        en += pick(", case insensitive", ", ignore case", ", any case", ", case doesn't matter")
    if R.random() < 0.15:
        en = en.lower()
    return engine.literal(rx, flags), en


def main():
    import bench
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R.seed(a.seed)
    held = {en.lower() for en, _, _ in bench.B}  # never emit a benchmark request verbatim
    seen, rows = set(), []
    for _ in range(a.n * 20):
        if len(rows) >= a.n:
            break
        rx, en = sample()
        if (rx, en) in seen or en.lower() in held:
            continue
        p, f = engine.parse_literal(rx)
        assert engine.compiles(p, f), rx
        seen.add((rx, en))
        rows.append({"en": en, "re": rx, "style": 6})
    with open(ROOT / "data/common.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), "pairs,", len({r["re"] for r in rows}), "regexes")


if __name__ == "__main__":
    main()
