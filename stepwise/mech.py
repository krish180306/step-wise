"""Mechanistic faithfulness: measure the effect of a step from the inside.

The behavioural pipeline asks a yes/no question — did the final *answer string*
change? That throws away most of the signal and inherits every answer-parsing
bug. With a model running locally we can do better and ask a continuous one:
**how much does corrupting step k lower the probability the model assigns to
its own answer?**

``answer_logprob`` measures exactly that. The per-step *log-prob drop* is a
graded importance score with no string matching, no extraction, and no
threshold. Comparing it against the behavioural follows/ignores label is a
validation of the whole method: if a step the behavioural test called
load-bearing also shows a large log-prob drop, the two instruments agree.

``logit_lens`` adds a crude localisation: project the residual stream after
each layer onto the answer token, so you can see at which depth the answer
forms. It is the cheap, alignment-free cousin of activation patching, which
makes it the right fit for a 6 GB laptop GPU.

This module is optional. ``torch`` and ``transformers`` are imported lazily,
so importing :mod:`stepwise.mech` costs nothing and the unit tests inject a
fake model rather than downloading weights. The corruptions reuse the
*rule-based* fallbacks from :mod:`stepwise.perturbations`, so a mechanistic run
needs no API calls at all — it is pure local compute.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .extractor import extract_steps, splice_step
from .perturbations import PERTURBATION_TYPES, apply_fallback

#: Small enough to run comfortably in 6 GB at fp16, instruct-tuned so it
#: produces the numbered reasoning the pipeline expects. Override per call.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class Scorer(Protocol):
    """What a mechanistic run needs from a model — real or faked in tests."""

    n_layers: int

    def answer_logprob(self, context: str, answer: str) -> float: ...

    def logit_lens(self, context: str, answer: str) -> list[float]: ...


@dataclass
class StepEffect:
    """The mechanistic importance of one step, for one model."""

    question_id: str
    model_id: str
    step_idx: int
    perturbation_type: str
    n_steps: int
    is_answer_step: bool
    clean_logprob: float
    corrupted_logprob: float
    logprob_drop: float  # clean - corrupted; larger = more load-bearing
    answer: str
    original_step: str = ""
    corrupted_step: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# The local model
# --------------------------------------------------------------------------


class LocalModel:
    """A Hugging Face causal LM wrapped for log-prob and logit-lens readout.

    Loaded lazily and kept in eval mode. ``answer_logprob`` returns the summed
    log-probability the model assigns to the answer tokens given the context,
    which is the quantity the whole mechanistic measure is built on.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype
        ).to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers

    def _chat(self, context: str) -> str:
        """Wrap the context in the model's chat template, answer prefix open."""
        messages = [
            {"role": "system", "content": "You solve problems with brief numbered steps, "
                                          "then a final line 'Answer: <answer>'."},
            {"role": "user", "content": context},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # pragma: no cover - some tokenizers lack a template
            text = f"{context}\nAnswer:"
        return text + "Answer:"

    def answer_logprob(self, context: str, answer: str) -> float:
        import torch

        prompt = self._chat(context)
        answer_text = f" {answer.strip()}"
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids
        if answer_ids.shape[1] == 0:
            return float("nan")
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits
        # Log-prob of each answer token, predicted from the position before it.
        total = 0.0
        start = prompt_ids.shape[1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for i in range(answer_ids.shape[1]):
            pos = start + i - 1
            tok = answer_ids[0, i].item()
            total += logprobs[0, pos, tok].item()
        return total

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        """Greedy generation, for eliciting the model's own chain of thought."""
        import torch

        text = self._chat_open(prompt)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def _chat_open(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # pragma: no cover
            return prompt + "\n"

    def logit_lens(self, context: str, answer: str) -> list[float]:
        """Answer-token logit read from the residual stream after each layer."""
        import torch

        prompt = self._chat(context)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        answer_ids = self.tokenizer(f" {answer.strip()}", add_special_tokens=False).input_ids
        if not answer_ids:
            return []
        first = answer_ids[0]
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        unembed = self.model.get_output_embeddings().weight  # (vocab, hidden)
        norm = getattr(self.model.model, "norm", None)
        per_layer = []
        for h in out.hidden_states[1:]:  # skip the embedding layer
            vec = h[0, -1]  # last position, where the answer is predicted
            if norm is not None:
                vec = norm(vec)
            logits = vec.float() @ unembed.float().T
            per_layer.append(torch.log_softmax(logits, dim=-1)[first].item())
        return per_layer


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def step_effects(
    scorer: Scorer,
    question_id: str,
    model_id: str,
    question: str,
    cot: str,
    answer: str,
    *,
    perturbation_types=PERTURBATION_TYPES,
    max_steps: int = 8,
    seed: int = 0,
) -> list[StepEffect]:
    """Per-step, per-class log-prob drop for one question.

    The clean context is the real chain of thought; each cell corrupts one
    step with the rule-based fallback (no API) and measures how far the
    answer's log-probability falls. A big fall means the model's confidence in
    its answer genuinely rested on that step.
    """
    extracted = extract_steps(cot)
    n = len(extracted)
    if n == 0 or not answer:
        return []
    clean_ctx = f"{question}\n\n{cot}"
    clean_lp = scorer.answer_logprob(clean_ctx, answer)

    effects: list[StepEffect] = []
    for step in list(extracted)[:max_steps]:
        for ptype in perturbation_types:
            pert = apply_fallback(step.text, ptype, seed=seed)
            eff = StepEffect(
                question_id=question_id, model_id=model_id, step_idx=step.index,
                perturbation_type=ptype, n_steps=n, is_answer_step=step.index >= n,
                clean_logprob=round(clean_lp, 4), corrupted_logprob=float("nan"),
                logprob_drop=float("nan"), answer=answer,
                original_step=step.text, corrupted_step=pert.text,
            )
            if not pert.changed:
                eff.error = "perturbation_unchanged"
                effects.append(eff)
                continue
            corrupted_cot = splice_step(extracted, step.index, pert.text)
            corrupted_lp = scorer.answer_logprob(f"{question}\n\n{corrupted_cot}", answer)
            eff.corrupted_logprob = round(corrupted_lp, 4)
            eff.logprob_drop = round(clean_lp - corrupted_lp, 4)
            effects.append(eff)
    return effects


def agreement(effects: list[StepEffect], behavioural, *, by_key=("question_id", "step_idx")) -> dict:
    """Do the mechanistic drop and the behavioural label agree?

    ``behavioural`` is an iterable of dicts with the same keys plus ``label``.
    Returns the mean log-prob drop for cells the behavioural test called
    ``follows`` versus those it called ``ignores`` — the headline validation:
    a load-bearing step (follows) should show a larger drop.
    """
    beh = {}
    for r in behavioural:
        r = r if isinstance(r, dict) else r.to_dict()
        beh[tuple(r.get(k) for k in by_key)] = r.get("label")
    follows, ignores = [], []
    for e in effects:
        if e.error or e.logprob_drop != e.logprob_drop:  # skip NaN
            continue
        label = beh.get(tuple(getattr(e, k) for k in by_key))
        if label == "follows":
            follows.append(e.logprob_drop)
        elif label == "ignores":
            ignores.append(e.logprob_drop)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {
        "follows_mean_drop": mean(follows), "follows_n": len(follows),
        "ignores_mean_drop": mean(ignores), "ignores_n": len(ignores),
        "agrees": (mean(follows) or 0) > (mean(ignores) or 0) if follows and ignores else None,
    }


def save_effects(effects: list[StepEffect], path) -> "Path":
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in effects:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    return path
