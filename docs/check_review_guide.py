#!/usr/bin/env python
"""Check (or repair) the `file.py:NNN` references in REVIEW_GUIDE.md.

The guide answers a reviewer's questions with the exact line behind every
claim, which is its whole value — and which rots the moment anything above
those lines changes. A stale pointer is worse than no pointer: it sends a
reader to a plausible but wrong function.

This checks every reference that names its symbol, as in::

    `pipeline.py:580` `run_evaluation`

by parsing the sources for that symbol's real line. References with no symbol
beside them are counted and listed but never guessed at — inferring one from
nearby prose crosses table rows and produces confidently wrong numbers.

    python docs/check_review_guide.py          # report, exit 1 if anything drifted
    python docs/check_review_guide.py --fix    # rewrite the drifted ones in place

Run it after editing anything the guide points at; CI can run the check form
to keep the guide honest.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUIDE = ROOT / "REVIEW_GUIDE.md"

#: A reference followed immediately by the symbol it points at. The symbol has
#: to be the very next backticked token (an arrow between them is allowed), so
#: that nothing further down the line can be mistaken for it.
REFERENCE = re.compile(
    r"`(?P<file>[\w/]+\.py):(?P<line>\d+)`"
    r"(?P<sep>[ \t]*(?:->|→)?[ \t]*)"
    r"`(?P<symbol>[A-Za-z_][\w.]*)`"
)
ANY_REFERENCE = re.compile(r"`[\w/]+\.py:\d+`")


def symbol_table() -> dict[tuple[str, str], int]:
    """Map (file basename, symbol) to the line it is defined on.

    Methods are recorded under both their bare name and ``Class.method``, so
    the guide can write either.
    """
    table: dict[tuple[str, str], int] = {}
    sources = [ROOT / "main.py"]
    sources += sorted(ROOT.glob("stepwise/*.py"))
    sources += sorted(ROOT.glob("stepwise/*/*.py"))
    for path in sources:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        def walk(node, class_name: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    table.setdefault((path.name, child.name), child.lineno)
                    if class_name:
                        table[(path.name, f"{class_name}.{child.name}")] = child.lineno
                    if isinstance(child, ast.ClassDef):
                        walk(child, child.name)

        walk(tree)
    return table


def resolve(table, basename: str, symbol: str) -> int | None:
    line = table.get((basename, symbol))
    if line is None and "." in symbol:
        line = table.get((basename, symbol.split(".")[-1]))
    return line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true",
                        help="rewrite drifted references instead of only reporting them")
    args = parser.parse_args(argv)

    if not GUIDE.exists():
        print(f"no guide at {GUIDE}")
        return 1

    table = symbol_table()
    text = GUIDE.read_text(encoding="utf-8")
    drifted: list[str] = []
    missing: list[str] = []
    correct = 0

    def rewrite(match: re.Match) -> str:
        nonlocal correct
        path = match.group("file")
        basename = path.split("/")[-1]
        stated = int(match.group("line"))
        symbol = match.group("symbol")
        actual = resolve(table, basename, symbol)
        if actual is None:
            missing.append(f"{path}:{stated} `{symbol}` - symbol not found")
            return match.group(0)
        if actual == stated:
            correct += 1
            return match.group(0)
        drifted.append(f"{path}:{stated} -> :{actual}  ({symbol})")
        if not args.fix:
            return match.group(0)
        return f"`{path}:{actual}`{match.group('sep')}`{symbol}`"

    updated = REFERENCE.sub(rewrite, text)
    bare = len(ANY_REFERENCE.findall(text)) - (correct + len(drifted) + len(missing))

    if args.fix and drifted:
        GUIDE.write_text(updated, encoding="utf-8")

    verb = "fixed" if args.fix else "drifted"
    print(f"{correct} correct, {len(drifted)} {verb}, {len(missing)} unresolvable, "
          f"{bare} bare (no symbol named, not checked)")
    for entry in drifted:
        print(f"  {'FIXED' if args.fix else 'DRIFT'} {entry}")
    for entry in missing:
        print(f"  ???   {entry}")

    if missing:
        print("\nAn unresolvable symbol usually means it was renamed or removed.")
    if bare:
        print(f"\n{bare} reference(s) name no symbol, so they cannot be checked "
              f"automatically.\nName the function beside them to bring them under "
              f"this check.")
    return 1 if (drifted and not args.fix) or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
