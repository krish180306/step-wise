"""The review guide's `file:line` references must point at the real lines.

The guide's value is that every claim carries the exact line behind it, which
rots silently as code moves. A stale pointer is worse than none: it sends a
reader to a plausible but wrong function. CI runs the same check.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKER = ROOT / "docs" / "check_review_guide.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_review_guide", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    if not CHECKER.exists():
        pytest.skip("checker script not present")
    return _load()


def test_the_guide_exists():
    assert (ROOT / "REVIEW_GUIDE.md").exists()


def test_every_named_reference_points_at_the_right_line(checker, capsys):
    # Exit 0 means no drifted and no unresolvable references.
    assert checker.main([]) == 0, capsys.readouterr().out


def test_the_symbol_table_finds_functions_and_methods(checker):
    table = checker.symbol_table()
    assert ("pipeline.py", "run_evaluation") in table
    assert ("pipeline.py", "RunResult.save") in table  # qualified
    assert ("pipeline.py", "save") in table            # and bare
    assert ("leakage.py", "answer_overwritten") in table
    assert ("stats.py", "required_questions") in table


def test_the_symbol_table_agrees_with_the_source(checker):
    table = checker.symbol_table()
    lines = (ROOT / "stepwise" / "leakage.py").read_text(encoding="utf-8").splitlines()
    line = table[("leakage.py", "step_states_answer")]
    assert lines[line - 1].startswith("def step_states_answer")


def test_resolve_falls_back_from_qualified_to_bare(checker):
    table = {("x.py", "save"): 42}
    assert checker.resolve(table, "x.py", "RunResult.save") == 42
    assert checker.resolve(table, "x.py", "nope") is None


def test_the_reference_pattern_requires_an_adjacent_symbol(checker):
    """Guessing from nearby prose crosses table rows and points at the wrong
    function, so only an immediately following symbol counts."""
    matched = checker.REFERENCE.search("`pipeline.py:580` `run_evaluation`")
    assert matched and matched.group("symbol") == "run_evaluation"
    assert checker.REFERENCE.search("`pipeline.py:580` `run_evaluation`").group("line") == "580"
    # An arrow between them is allowed.
    assert checker.REFERENCE.search("`a.py:1` -> `fn`").group("symbol") == "fn"
    # A bare reference is not matched, so it is never rewritten.
    assert checker.REFERENCE.search("`pipeline.py:621` in the pool") is None


def test_the_guide_documents_the_new_controls():
    text = (ROOT / "REVIEW_GUIDE.md").read_text(encoding="utf-8")
    assert "leakage.py" in text
    assert "step_deletion" in text
    assert "design effect" in text.lower()
