"""Hand-written everyday benchmark: request + strings that must / must not match.

Scored on behavior only (no gold regex). Negatives avoid ambiguous cases
(e.g. anchoring is only tested when the request implies it).

    python src/bench.py --ckpt ckpt/best
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

B = [
    ("match an email address", ["jo.doe@gmail.com", "a_b@site.org"], ["jo.doe", "@gmail.com", "jo@", "jo doe@gmail.com"]),
    ("us phone number like (555) 123-4567", ["(555) 123-4567"], ["555-1234", "(55) 123-4567", "phone"]),
    ("phone number like 555-123-4567", ["555-123-4567"], ["555-12-4567", "5551-23-4567", "abc-def-ghij"]),
    ("a url starting with http or https", ["http://example.com", "https://a.io/x"], ["ftp://example.com", "example.com", "htp://a.com"]),
    ("date in the format mm/dd/yyyy", ["03/14/2024", "12/01/1999"], ["2024-03-14", "3/14/24", "03-14-2024"]),
    ("iso date like 2024-01-31", ["2024-01-31", "1999-12-01"], ["01/31/2024", "2024-1-31", "24-01-31"]),
    ("a hex color code like #fff or #a1b2c3", ["#fff", "#a1b2c3"], ["fff", "#ggg", "#a1b2c", "#a1b2c3d4"]),
    ("password at least 8 characters with only letters and numbers", ["hunter2024", "abcdefgh"], ["short1", "has space1", "p@ssword!"]),
    ("only digits", ["123", "0"], ["12a", "", "1.5"]),
    ("only letters", ["abc", "Hello"], ["abc1", "", "a b"]),
    ("lowercase letters only", ["abc"], ["Abc", "ab1", ""]),
    ("exactly 5 digits", ["12345", "00000"], ["1234", "123456", "12a45"]),
    ("five digit zip code", ["90210"], ["9021", "902100", "9o210"]),
    ("us zip code with optional +4", ["90210", "90210-1234"], ["9021", "90210-12", "90210 1234"]),
    ("lines that end with a semicolon", ["x = 1;", ";"], ["x = 1", "x = 1.", "x = 1:"]),
    ("starts with a capital letter", ["Hello", "A"], ["hello", "1abc", " Hello"]),
    ("ends with a question mark", ["why?", "?"], ["why", "why.", "why!"]),
    ("a dollar amount like $12.99", ["$12.99", "$0.50"], ["12.99", "$12.9", "$ab.cd"]),
    ("ipv4 address", ["192.168.0.1", "8.8.8.8"], ["192.168.0", "192.168.0.1.5", "a.b.c.d"]),
    ("time in 24 hour format like 13:45", ["13:45", "00:00", "23:59"], ["1345", "13-45", "ab:cd"]),
    ("a hashtag", ["#regex", "#a1"], ["regex", "# space"]),
    ("a twitter handle like @nate", ["@nate", "@a_b1"], ["nate", "@", "@ nate"]),
    ("jpg or png file extension", ["photo.jpg", "a.png"], ["photo.gif", "photo.jpeg.txt", "jpg"]),
    ("a pdf file name", ["report.pdf", "a-b.pdf"], ["report.doc", "pdf", "report.pdf.exe"]),
    ("trailing whitespace", ["abc  ", "x\t"], ["abc", " abc"]),
    ("blank or whitespace-only line", ["", "   "], ["a", " a "]),
    ("contains the word error", ["an error occurred", "error"], ["no problems", "err"]),
    ("lines starting with TODO", ["TODO: fix", "TODO later"], ["fix TODO", "todo: fix"]),
    ("a uuid", ["123e4567-e89b-12d3-a456-426614174000"], ["123e4567e89b12d3a456426614174000", "not-a-uuid"]),
    ("semantic version like 1.2.3", ["1.2.3", "10.0.12"], ["1.2", "1.2.x", "a.b.c"]),
    ("a percentage like 45%", ["45%", "100%", "5%"], ["45", "%45", "4 5%"]),
    ("a positive integer with no leading zeros", ["1", "42", "1000"], ["0", "042", "-1", "4.2"]),
    ("a decimal number like 3.14", ["3.14", "0.5"], ["3", "3.", ".", "3.1.4"]),
    ("username 3 to 16 characters, letters numbers underscores", ["nate_99", "abc"], ["ab", "a" * 17, "na te", "nate!"]),
    ("binary string of 0s and 1s", ["0101", "1"], ["0102", "", "10 1"]),
    ("a word that ends in ing", ["running", "sing"], ["runs", "ingot"]),
    ("credit card number, 16 digits", ["1234567812345678"], ["123456781234567", "12345678123456789", "1234-5678-1234-567a"]),
    ("a us state abbreviation, two capital letters", ["CA", "NY"], ["ca", "C", "CAL"]),
    ("an html tag like <div>", ["<div>", "<p>"], ["div", "<>", "< div"]),
    ("text in double quotes", ['say "hello"', '"x"'], ["say hello", "'x'"]),
]


# second set: written after common.py, different wording, never used to tune anything
B2 = [
    ("validate an email", ["me@x.com", "first.last@company.io"], ["me@x", "me.x.com", "@x.com"]),
    ("check if the input is all numbers", ["2024", "7"], ["20a4", "", "12 34"]),
    ("string has to be 10 digits long", ["0123456789"], ["012345678", "01234567890", "012345678a"]),
    ("starts with https://", ["https://a.com", "https://"], ["http://a.com", "xhttps://a.com"]),
    ("anything ending in .txt", ["notes.txt", "a.b.txt"], ["notes.txt.bak", "notestxt", "notes.md"]),
    ("must be between 4 and 8 characters", ["abcd", "abcdefgh"], ["abc", "abcdefghi"]),
    ("a 24 hour clock time hh:mm", ["09:30", "23:59"], ["9:3", "0930", "ab:cd"]),
    ("lines beginning with a # comment", ["# hi", "#x"], ["hi #", " x"]),
    ("a negative or positive whole number", ["-5", "42", "0"], ["4.2", "--5", "5-"]),
    ("only uppercase letters and digits", ["AB12", "Z"], ["ab12", "AB-12", ""]),
    ("a word starting with un", ["undo", "unhappy"], ["fun", "run"]),
    ("three letters followed by three digits like ABC123", ["ABC123", "xyz789"], ["AB123", "ABC12", "123ABC"]),
    ("file names ending in .js or .ts", ["app.js", "index.ts"], ["app.jsx", "app.json", "appjs"]),
    ("a year from 1900 to 2099", ["1999", "2024", "1900"], ["1899", "2100", "99"]),
    ("whitespace at the start of a line", ["  x", "\tx"], ["x", "x  "]),
    ("contains a digit", ["abc1", "9"], ["abc", ""]),
    ("a mac address like 01:23:45:67:89:ab", ["01:23:45:67:89:ab", "AA:BB:CC:DD:EE:FF"], ["01:23:45:67:89", "0123456789ab", "gg:23:45:67:89:ab"]),
    ("the word cat or dog", ["cat", "dog"], ["cow", "ca"]),
    ("a string with no spaces", ["abc", "a-b"], ["a b", " "]),
    ("price in dollars and cents like $3.50", ["$3.50", "$100.00"], ["3.50", "$3.5", "$3"]),
]


def run(model, k=8, hints=True, bench=None):
    import engine
    import infer
    rows = []
    for en, pos, neg in bench or B:
        c = infer.best(model, en, k=k, use_hints=hints)
        pred = c[0][0] if c else ""
        try:
            p, f = engine.parse_literal(pred)
            ok = engine.test(p, f, pos) == [True] * len(pos) and engine.test(p, f, neg) == [False] * len(neg)
        except Exception:
            ok = False
        rows.append({"en": en, "pred": pred, "ok": ok})
    return rows


def main():
    import infer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ROOT / "ckpt/best")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--set", type=int, default=1, help="1 = original 40, 2 = fresh held-out 20")
    a = ap.parse_args()
    rows = run(infer.load(a.ckpt, a.bits), hints=not a.no_hints, bench=B if a.set == 1 else B2)
    if a.v:
        for r in rows:
            print(f"{'✓' if r['ok'] else '✗'} {r['en']:62} {r['pred']}")
    print(json.dumps({"everyday": sum(r["ok"] for r in rows) / len(rows), "n": len(rows)}))


if __name__ == "__main__":
    main()
