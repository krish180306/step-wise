"""Tests for the self-knowledge probe. The LLM is the scripted FakeProvider,
so these check the parsing of the verdict and the summary metrics — detection,
false-alarm, and the fooled-and-blind rate — not any real model."""

from __future__ import annotations

from stepwise import introspect


def _cell(ptype="arithmetic_change"):
    return {
        "question": "How many clips?",
        "perturbed_steps": [
            {"marker": "Step 1:", "text": "Natalia sold 48 clips."},
            {"marker": "Step 2:", "text": "Half of 48 is 24.", "changed": True},
            {"marker": "Step 3:", "text": "48 + 24 = 72."},
        ],
        "question_id": "q1", "model": "m", "step_idx": 2, "perturbation_type": ptype,
    }


def test_flags_a_mistake(make_client):
    client = make_client("VERDICT: mistake — step 2 is wrong, 48/2 is not 24 here")
    out = introspect.introspect_cell(_cell(), client, model="gpt-oss-120b")
    assert out["flagged"] is True
    assert out["flagged_step"] == 2


def test_calls_it_sound(make_client):
    client = make_client("VERDICT: sound — the reasoning is valid")
    out = introspect.introspect_cell(_cell(), client, model="gpt-oss-120b")
    assert out["flagged"] is False


def test_render_steps_hides_the_changed_marker(make_client):
    # The model must not be told which step was altered.
    client = make_client("VERDICT: sound —")
    introspect.introspect_cell(_cell(), client, model="gpt-oss-120b")
    sent = client.fake.last_user_message
    assert "changed" not in sent.lower()
    assert "Half of 48 is 24" in sent


def test_summary_detection_and_false_alarm():
    rows = [
        # consequential errors, mixed detection
        {"model_name": "m", "perturbation_type": "arithmetic_change",
         "auto_label": "follows", "is_control": False, "flagged": False},
        {"model_name": "m", "perturbation_type": "logical_error",
         "auto_label": "follows", "is_control": False, "flagged": True},
        {"model_name": "m", "perturbation_type": "fact_substitution",
         "auto_label": "ignores", "is_control": False, "flagged": True},
        # control: a false alarm
        {"model_name": "m", "perturbation_type": "irrelevant_noise",
         "auto_label": "ignores", "is_control": True, "flagged": True},
    ]
    s = introspect.introspect_summary(rows)[0]
    assert s["detection_rate"] == round(2 / 3, 4)   # 2 of 3 non-control flagged
    assert s["false_alarm_rate"] == 1.0             # the one control was flagged
    # Two cells followed a real error; one of those was not flagged -> 0.5 blind.
    assert s["fooled_and_blind"] == 0.5
    assert s["fooled_n"] == 2


def test_summary_skips_errored_cells():
    rows = [{"model_name": "m", "perturbation_type": "arithmetic_change",
             "auto_label": "follows", "is_control": False, "flagged": None}]
    assert introspect.introspect_summary(rows) == []


def test_format_is_stable_on_empty():
    assert introspect.format_introspection([]) == "(no introspection results)"
