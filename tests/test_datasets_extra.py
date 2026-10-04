"""The A5 additions: SVAMP, AQuA, 2WikiMultiHop loaders (via fixtures, no
network) and their domain tagging, which is what makes the arithmetic and
multi-hop domains have more than one source each."""

from __future__ import annotations

from stepwise.datasets import load_fixtures, _tag_domains, _normalise_number, ANSWER_TYPES
from stepwise.domains import domain_for, ARITHMETIC, MULTIHOP


def test_new_datasets_have_fixtures_and_types():
    for name, atype in (("svamp", "numeric"), ("aqua", "label"), ("2wikimultihop", "span")):
        samples = load_fixtures(name)
        assert samples, f"{name} has no fixtures"
        assert ANSWER_TYPES[name] == atype
        assert all(s.dataset == name for s in samples)


def test_domains_now_have_two_sources():
    # Arithmetic: gsm8k + svamp + aqua; multi-hop: hotpotqa + 2wikimultihop.
    assert domain_for("svamp") == ARITHMETIC
    assert domain_for("aqua") == ARITHMETIC
    assert domain_for("2wikimultihop") == MULTIHOP
    arithmetic_sources = [d for d in ("gsm8k", "svamp", "aqua") if domain_for(d) == ARITHMETIC]
    assert len(arithmetic_sources) >= 2


def test_tag_domains_fills_domain():
    samples = _tag_domains(load_fixtures("svamp"))
    assert all(s.domain == ARITHMETIC for s in samples)


def test_aqua_choices_are_lettered():
    s = load_fixtures("aqua")[0]
    assert s.choices and all(c[1:3] == ". " for c in s.choices)
    assert s.gold_answer in {c[0] for c in s.choices}


def test_normalise_number():
    assert _normalise_number("51.0") == "51"
    assert _normalise_number("204") == "204"
    assert _normalise_number("1,000") == "1000"
    assert _normalise_number("3.5") == "3.5"
