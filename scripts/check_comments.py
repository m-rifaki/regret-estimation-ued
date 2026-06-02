"""Reject comments that read as though a machine wrote them.

Run it over the package and it prints one line per suspect comment. It checks four things a
human writing at the keyboard does not do: reach for a stock adjective, restate the line below,
run three clauses together for rhythm, or spell a word the British way.

  uv run python scripts/check_comments.py
"""
from __future__ import annotations

import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).parent.parent

# words that mark generated prose, and the words the author has banned by name
STOCK = [
    "delve", "leverage", "utiliz", "robustly", "seamless", "comprehensive", "crucial",
    "essential", "powerful", "elegant", "simply", "basically", "in essence", "worth noting",
    "note that", "furthermore", "moreover", "additionally", "we can see", "this ensures",
    "allows us to", "facilitates", "framework", "load-bearing", "wedge",
]
BRITISH = r"colour|behaviour|analyse[ds]?\b|normalis|labelled|modelling|grey\b|towards|whilst"

RULES = [
    ("stock phrase", re.compile(r"\b(" + "|".join(STOCK) + r")", re.I)),
    ("British spelling", re.compile(BRITISH, re.I)),
    ("em dash", re.compile("\\u2014")),
]


def comments(path: Path):
    """Every comment and docstring in a file, as (line number, text)."""
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string.lstrip("# ").rstrip()
            elif tok.type == tokenize.STRING and tok.line.lstrip().startswith(('"""', "'''")):
                yield tok.start[0], tok.string.strip("\"'")


PRAGMA = re.compile(r"^(type:|noqa|ruff:|pylint:|pragma:|fmt:|isort:)")


def prose(text: str) -> str:
    """The part of a comment that is prose. A call, a subscript, a run of numbers and a list of
    short names are all data, so none of them counts toward the one-comma rule."""
    text = re.sub(r"^[-=# ]+$", " ", text)                    # a section divider carries no prose
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text)          # a call or a subscript is code
    text = re.sub(r"[-0-9.]+(\s*,\s*[-0-9.]+)+", " ", text)   # a run of numbers is data
    # a comma-separated run of short names is a list of things and a list is allowed its commas
    return re.sub(r"(\b[\w.]+(?:\s+[\w.]+){0,2}\s*,\s*){2,}", " ", text)


def restates(text: str, following: str) -> bool:
    """A comment whose every word is already an identifier on the next line says nothing."""
    words = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in ("with", "from", "that")]
    if len(words) < 2:
        return False
    return all(w in following.lower() for w in words)


def main() -> int:
    bad = []
    trees = ("src", "experiments", "tests", "scripts")
    for path in sorted(q for t in trees for q in ROOT.glob(f"{t}/**/*.py")):
        lines = path.read_text().splitlines()
        for lineno, text in comments(path):
            rel = path.relative_to(ROOT)
            if PRAGMA.match(text):
                continue
            for name, pattern in RULES:
                hit = pattern.search(text)
                if hit:
                    bad.append(f"{rel}:{lineno}  {name} {hit.group(0)!r}: {text[:60]}")
            if prose(text).count(",") > 1 and len(text) < 200:
                bad.append(f"{rel}:{lineno}  more than one comma: {text[:60]}")
            if lineno < len(lines) and restates(text, lines[lineno]):
                bad.append(f"{rel}:{lineno}  restates the line: {text[:70]}")
            if re.match(r"^(Args|Returns|Parameters|Raises):", text.strip()):
                bad.append(f"{rel}:{lineno}  signature boilerplate: {text[:70]}")
    for line in bad:
        print(" ", line)
    print(f"\n{len(bad)} suspect comments")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
