"""Pull example strings out of a request, to check candidate regexes against.

    "a hex color like #fff or #a1b2c3"  -> ["#fff", "#a1b2c3"]
    "date in the format mm/dd/yyyy"     -> ["01/31/2024"]
    'starts with "foo"'                 -> ["foo..."]  (quoted text is not an example; skipped)
"""
import re

CUE = re.compile(r"\b(?:like|e\.g\.?|eg|such as|for example|for instance|ex|example)\s*[:,]?\s+", re.I)
STOP = re.compile(r"\s+(?:or|and|,)\s+|\s*,\s*|\s*;\s*")
END = re.compile(r"\s+(?:with|that|where|which|but|in|at|from|to|for|ignoring|case|optional|nothing|anywhere)\b.*$", re.I)
# date/time format strings -> a concrete instance
FMT = re.compile(r"\b(?=[ymdhs:/. -]*[ymd])[ymdhs]{1,4}(?:[-/.: ][ymdhs]{1,4}){1,5}\b", re.I)
PARTS = {"yyyy": "2024", "yy": "24", "mm": "01", "m": "1", "dd": "31", "d": "9", "hh": "12", "h": "7", "ss": "59", "s": "5"}


def _fmt(s):
    out = []
    for tok in re.split(r"([-/.: ])", s):
        if tok in "-/.: " and tok:
            out.append(tok)
        else:
            v = PARTS.get(tok.lower())
            if v is None:
                return None
            out.append(v)
    return "".join(out)


def _looks_concrete(s):
    """An example value, not a description: has a digit or symbol, or is one short token."""
    return bool(s) and len(s) <= 40 and (re.search(r"[\d#@$%:/.\-_()]", s) is not None) and " " not in s.strip("() ") or bool(re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", s))


def examples(en: str) -> list[str]:
    out = []
    for m in CUE.finditer(en):
        tail = END.sub("", en[m.end():])
        for chunk in STOP.split(tail)[:4]:
            c = chunk.strip().strip("'\"`").rstrip(".?!")
            if _looks_concrete(c):
                out.append(c)
    for m in FMT.finditer(en):
        if not re.search(r"yy|mm|dd|hh", m.group(0), re.I):  # "m:h", "b d h w" are not formats
            continue
        v = _fmt(m.group(0))
        if v:
            out.append(v)
    return list(dict.fromkeys(out))


if __name__ == "__main__":
    for q in ["a hex color code like #fff or #a1b2c3", "date in the format mm/dd/yyyy",
              "us phone number like (555) 123-4567", "a dollar amount like $12.99",
              "version number like v1.2.3", "iso date like 2024-01-31", "match an email address",
              "time like 13:45 or 9:05", "an id such as AB-1234, case insensitive"]:
        print(f"{q:45} {examples(q)}")
