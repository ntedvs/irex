"""JavaScript regex semantics via embedded QuickJS."""
import json
import quickjs

_JS = r"""
function compiles(p, f) { try { new RegExp(p, f); return true } catch (e) { return false } }
function tests(p, f, xs) { const r = new RegExp(p, f); return JSON.stringify(JSON.parse(xs).map(s => r.test(s))) }
"""

_ctx = quickjs.Context()
_ctx.set_time_limit(0.05)  # seconds; guards against catastrophic backtracking
_ctx.eval(_JS)
_compiles, _tests = _ctx.get("compiles"), _ctx.get("tests")


def compiles(pattern: str, flags: str = "") -> bool:
    return bool(_compiles(pattern, flags))


def test(pattern: str, flags: str, strings: list[str]) -> list[bool]:
    return json.loads(_tests(pattern, flags, json.dumps(strings)))


def literal(pattern: str, flags: str = "") -> str:
    return f"/{pattern}/{flags}"


def parse_literal(s: str) -> tuple[str, str]:
    s = s.strip()
    if not s.startswith("/") or s.rfind("/") == 0:
        raise ValueError(f"not a regex literal: {s!r}")
    i = s.rfind("/")
    return s[1:i], s[i + 1:]
