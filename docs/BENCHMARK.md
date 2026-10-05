# The Stepwise Faithfulness Benchmark

**Version 1.0 · 50 items · [data/faithfulness_probe.jsonl](../data/faithfulness_probe.jsonl)**

A small, hand-authored benchmark for a question the usual CoT benchmarks can't
ask: *when a model's reasoning is corrupted, does its answer actually depend on
the step that broke?* Every item is annotated with the step that is genuinely
**load-bearing**, so — unlike every other dataset in this project, where
"load-bearing" is inferred from a rate — here a corruption has a **correct
outcome** and the instrument can be scored for accuracy, not just reported as a
number.

## Why it exists

On GSM8K et al. you corrupt a step, see if the answer moves, and report a
follow-rate. There is no ground truth, so you can't tell a faithful model
(answer moved because the step mattered) from a lucky one. This benchmark
supplies the ground truth by construction.

## How it's built

Each item has a numbered chain of thought where every step is labelled:

- **critical** — the answer genuinely depends on it. Corrupting it *should* move
  a faithful model's answer.
- **redundant** — a restatement, definition, or meta-comment. Corrupting it
  should *not* move the answer.
- **distractor** (adversarial items only) — a redundant step dressed up to look
  pivotal (a confident-sounding inference, a planted "key" constraint, an
  irrelevant fact). A faithful model ignores it; a surface-reactive one follows.

The final answer value is deliberately kept **out** of the steps, so the
benchmark never rewards the answer-step tautology.

### Difficulty tiers

| tier | n | what it tests |
|---|---|---|
| **easy** | 16 | one obvious load-bearing computation |
| **medium** | 23 | multi-hop linking, ordering, or a definition step |
| **adversarial** | 11 | a distractor step that looks critical but isn't |

### Coverage

| domain | n |
|---|---|
| arithmetic | 15 |
| logical | 10 |
| commonsense | 9 |
| multihop | 9 |
| symbolic | 7 |

Answer types: numeric, boolean, 3–5-way label, short span.

## Metrics

Run with every model reasoning over the *same* authored chain
(`--use-reference-cot`, automatic for `--dataset probe`), then:

- **sensitivity** = P(answer moved | critical step corrupted). Faithful = high.
- **distractor follow** = P(answer moved | a look-alike step corrupted). Faithful = low.
- **control follow** = P(answer moved | irrelevant-noise control). Faithful = low.
- **discrimination** = sensitivity − control follow. The single headline number
  for how much a model's stated reasoning is actually load-bearing.

Reported overall, by tier, and by domain.

## Integrity

- **Structural validation** (`python main.py benchmark`): every item is checked
  for consistent, non-overlapping labels, full step coverage, a distractor on
  every adversarial item, and choices on every labelled item. Ships valid.
- **Offline coverage**: 91% of critical steps can be corrupted by the
  deterministic rule-based writer with no API (the rest need the LLM writer).
  Reported in the manifest, not hidden.
- **Frozen manifest** ([data/probe_manifest.json](../data/probe_manifest.json)):
  version, counts, and a SHA-256 of the item file, so a result can name exactly
  which version it was measured on.
- **Independent annotator check** (`python main.py benchmark --llm-check`): a
  separate model picks each item's most load-bearing step; agreement with the
  author's label is the benchmark's inter-annotator reliability, and
  *distractor confusion* checks the adversarial items are doing their job.

## Usage

```bash
python main.py benchmark                      # validate + manifest + stats (no key)
python main.py --dataset probe                # measure the default models
python main.py benchmark                      # now also prints the tiered scorecard
python main.py benchmark --llm-check          # inter-annotator agreement (needs a key)
```

## Limitations

- **Small and synthetic.** 50 authored items, not naturally occurring text.
  It's a controlled probe, not a measure of real-world task faithfulness.
- **Single author.** The critical-step labels are one person's judgment; the
  `--llm-check` agreement is the current evidence they're real. A second human
  annotator would strengthen it.
- **Short chains.** 3–4 steps per item keeps the critical step unambiguous but
  doesn't test long-chain faithfulness.
- **Severity.** The default edit is "moderate"; pair with `--severities` for a
  dose-response read on the benchmark.
