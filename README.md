# Stepwise

**Is a model's chain of thought load-bearing, or is it a story told after the
fact?** Stepwise answers that with five causal interventions, a mechanistic
cross-check, validated measurement, and an agentic analysis layer — over open
models on the Groq free tier.

A chain of thought is only an explanation if the answer depends on it. Stepwise
tests that dependence directly: it corrupts a model's *own* reasoning, re-runs
the model, and measures what the answer does — behaviourally (does the answer
move?) and mechanistically (does the probability of the answer fall?).

```bash
pip install -r requirements.txt
cp .env.example .env          # add a free key from https://console.groq.com/keys

python main.py demo                                 # offline, no key
python main.py --dataset gsm8k --n_samples 50       # step-corruption run
python main.py --dataset probe                      # faithfulness probe set (KNOWN critical steps)
python main.py --dataset gsm8k --severities minimal,moderate,severe   # dose-response ladder
python main.py --dataset gsm8k --writer-model allam-2-7b   # independent perturbation writer (de-confound)
python main.py battery --dataset gsm8k              # the 5-test battery
python main.py mech --dataset gsm8k --validate --patch     # mechanistic track: same-model check + causal tracing
python main.py introspect --dataset gsm8k          # self-knowledge probe: does the model notice?
python main.py analyse --dataset gsm8k              # agent writes a report, then fact-checks it
python main.py validate --dataset gsm8k             # label reliability + regression
python main.py dashboard                            # http://127.0.0.1:8000
python -m pytest tests/ -q                          # 409 tests, no network, no GPU
```

---

## The headline finding (on the current 2-question run)

The project's own data already shows why the naive metric is misleading, and
three independent methods agree:

| method | what it found |
|---|---|
| **answer-step control** | 100% of gpt-oss-120b's and qwen3.6's apparent "dependency" came from corrupting the *answer-stating* final step — a tautology, not reasoning |
| **early answering** | gpt-oss-120b gave the correct answer with **zero** reasoning shown |
| **mechanistic** | corrupting the answer step drops the answer's log-probability by **+2.3**; corrupting reasoning steps, **~−0.2** |

Two questions is a working pipeline, not a result. But the instrument now
separates real reasoning-dependence from the trivial case that the original
headline number hid.

---

## What's in it

### 1. Step corruption — the original probe
Corrupt exactly one step (fact / logic / arithmetic / irrelevant-noise
control), re-run, label **follows / ignores / corrects**. The faithfulness
pipeline: `pipeline.py`, `perturbations.py`, `extractor.py` (lossless
segmentation), `answers.py`.

### 2. The battery — four more tests (`battery.py`)
One probe can't separate the ways a CoT fails, so four more triangulate it
(Lanham 2023; Turpin 2023):

| test | detects |
|---|---|
| **early answering** | did the model decide before reasoning? |
| **paraphrase** | surface-form sensitivity (answer *should not* move) |
| **filler** | was the CoT computing, or just buying forward passes? |
| **biased context** | does the answer flip to a planted hint while the CoT stays silent — post-hoc rationalisation? |

### 3. Mechanistic track — from the inside (`mech.py`)
Runs a small model locally (Qwen2.5-0.5B on a 6 GB GPU) and measures a
**continuous** quantity: how much corrupting a step lowers the log-probability
of the answer. No string matching, no extraction bugs. Three things here:
- **`--validate`** runs the behavioural probe on the *same* local model, so the
  mechanistic↔behavioural agreement is **within one model** — the only version
  that actually validates the measure (earlier it compared the local model's
  drops to the Groq models' labels, which is not a fair test).
- **`--patch`** is **true activation patching**: patch the clean residual
  stream back into the corrupted run one layer at a time and watch how much of
  the answer is restored. That is *causal* localisation — which layer carries
  the corrupted step's effect — not the correlational drop. A per-layer
  restoration curve lands in the dashboard.
- Pure local compute, no API calls.

### 4. Domains, chance correction, honest intervals (`domains.py`, `stats.py`)
- **Domains** above datasets (BBH mapped per subtask), so "is reasoning more
  load-bearing in math than commonsense?" is a testable claim.
- **Chance correction** removes the floor that makes low-option answer formats
  (yes/no, MCQ) look more faithful for free — the bias that would otherwise
  confirm the hypothesis by itself.
- **Clustered bootstrap** CIs resample whole questions, since a question's cells
  aren't independent.

### 5. Measurement validation (`validation.py`, `judge.py`)
- **Cohen's κ** of the automatic label against human annotations (label cells
  in the dashboard) and against an **independent LLM judge** that reads the full
  drill-down.
- **Logistic regression** for P(follows) with cluster-robust SEs, separating
  step position from the answer-step confound. Flags perfect separation instead
  of printing nonsense on tiny data.

### 5a. The faithfulness probe set — ground truth (`data/faithfulness_probe.jsonl`, `probe_score.py`)
Everywhere else "load-bearing" is *inferred*: corrupt a step, see if the answer
moves, report a rate — but there's no ground truth to check the rate against.
The probe set supplies it. It's a hand-authored, in-repo set (22 items across
all domains) where each item's **critical** step (the one the answer genuinely
depends on) and **redundant** steps (restatements, meta-commentary) are
annotated. Every model reasons over the *same* authored chain
(`--use-reference-cot`, auto-on for `--dataset probe`), so the labels apply
uniformly, and a corruption now has a *correct* outcome:

- corrupt the **critical** step → a faithful model's answer **should move**;
- corrupt a **redundant** step or the **noise control** → it **should not**.

That makes it a real 2×2. `probe_score` reports **sensitivity** (moved when the
key step broke), **control follow-rate** (moved on a meaningless edit), and
**discrimination = sensitivity − control** — the single number for how much a
model's stated reasoning is actually doing the work, not a rate you have to take
on faith.

### 5c. Dose-response and capability (`perturbations.py`, `score.py`)
- **Severity ladder.** The same cell corrupted at three edit sizes — a
  near-miss (`minimal`), a clear error (`moderate`, the original), and an
  order-of-magnitude error (`severe`). `score.severity_curve` tracks follow-rate
  vs severity, split into the consequential classes and the noise control. A
  faithful model follows *more* as the real error grows while the control stays
  flat; a control line that climbs with the amount of noise is surface
  reactivity. (`moderate` is byte-identical to the pre-ladder behaviour, so old
  runs reproduce.)
- **Capability vs faithfulness.** `score.capability_vs_faithfulness` plots each
  model's baseline accuracy against how much its answer tracks its reasoning,
  with the Pearson correlation across models — does being smarter make a model's
  CoT more load-bearing, or is capability unrelated to faithfulness?

### 5b. Removing the confounds, and two new probes (`score.py`, `introspect.py`)
- **Writer de-confound.** A model editing its own reasoning is a confound.
  Every cell now records `self_perturbed` (writer == target), and
  `score.writer_effect` reports dependency split by writer. `--writer-model`
  picks an independent writer and removes it outright.
- **Cross-model transfer** (`score.cross_model_transfer`) asks whether a step
  that is brittle for one model is brittle for another — Jaccard overlap of the
  cells each model follows. Shared brittleness is a property of the question;
  model-specific brittleness is a property of the model.
- **Self-knowledge probe** (`introspect.py`): show a model its own chain with
  one step silently corrupted and ask whether it contains a mistake. Reports
  detection rate, the control's false-alarm rate, and the worst case —
  *fooled-and-blind*: followed the error into the answer and never noticed.

### 6. Agents + adversarial verifier (`agent/`)
- A **report agent** and a streaming **chat agent**, each a real tool-calling
  loop over ten query tools — they decide what to compute, and every number
  comes from a tool, never a guess.
- An **adversarial verifier** (`agent/verifier.py`) re-derives every number in
  a report from the records and flags the unsupported ones — including the exact
  bug the analyst once produced (a true p-value attached to the wrong
  comparison). Wired into every saved report.

### 7. MCP server (`mcp_server.py`)
The analysis tools exposed over MCP (stdio), so Claude Desktop or Claude Code
can query a run directly. Read-only, no Groq calls. See
`docs/mcp-client-config.json`.

### 8. Dashboard (`dashboard/`)
The findings surface — tautology gap, domain comparison with CIs, per-step
grid, the side-by-side explorer with inline human-annotation buttons, the agent
report, and the streaming Ask panel. FastAPI serves the API and the page as one
process.

---

## Models

Four Groq free-tier chat models (checked live): `gpt-oss-120b`, `qwen3.6-27b`,
`qwen3.8-27b`, `allam-2-7b` — three vendors, 7B to 120B. The Qwen models run
with thinking switched off (otherwise the `<think>` scratchpad leaks into the
reasoning). `python main.py models` lists them. The mechanistic track uses a
separate local HF model (default `Qwen/Qwen2.5-0.5B-Instruct`).

## Datasets & domains

GSM8K, **SVAMP**, **AQuA** (arithmetic), StrategyQA, CommonsenseQA
(commonsense), HotpotQA, **2WikiMultiHop** (multi-hop), BBH (one subtask per
run), loaded from HuggingFace onto one schema with an offline fixture fallback.
Each domain now has ≥2 dataset sources, so "is reasoning more load-bearing in
math than commonsense?" is a comparison across sources, not one dataset's
quirks.

## Caching & cost

Every Groq response is cached to `.cache/` keyed on
`(question_id, step_idx, perturbation_type, model)` plus text hashes, so an
interrupted run resumes for free and re-scoring costs nothing. The battery
shares the baseline call through the same cache. Start runs small — the free
tier is rate-limited and a full grid is thousands of calls.

## Testing, CI, Docker

```bash
pytest tests/ -q      # 409 tests, all offline (scripted provider, fake scorer)
```

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.11–3.13 with no key
and no GPU. The dashboard containerises with the provided `Dockerfile`:

```bash
docker build -t stepwise .
docker run -p 8000:8000 -v "$PWD/results:/app/results" stepwise
```

See `docs/DEPLOY.md` for a public Hugging Face Spaces deploy (run the live chat
on the visitor's own key, never yours).

## Layout

```
stepwise/
  pipeline.py        elicit → segment → corrupt → re-run → classify
  perturbations.py   the four corruptions (strict prompts + rule fallbacks)
  extractor.py       lossless CoT → steps, and splicing
  answers.py         answer extraction, normalisation, matching
  score.py           aggregation, chance correction, answer-step split, CIs
  stats.py           chance rate, clustered bootstrap
  domains.py         reasoning domains above datasets
  battery.py         early answering, paraphrase, filler, biased context
  battery_score.py   one metric per battery test
  mech.py            log-prob step effect, same-model probe, activation patching
  introspect.py      self-knowledge probe (does the model notice?)
  probe_score.py     sensitivity/control against the probe set's known critical steps
  validation.py      Cohen's κ, annotation store, regression
  judge.py           independent LLM judge
  datasets.py        nine loaders + fixtures (incl. the in-repo probe set)
  demo.py            offline simulated run
  mcp_server.py      tools over MCP (stdio)
  agent/
    tools.py         ten query tools
    analyst.py       report agent (+ auto verification)
    chat.py          streaming chat agent
    verifier.py      adversarial fact-checker
  dashboard/
    app.py           FastAPI API + SSE chat + annotation endpoints
    static/index.html the page
tests/               409 tests, no network
docs/                call map, deploy guide, MCP config
```

## Limitations

- **Scale.** The real finding needs 50+ questions across domains; the current
  numbers are from 2. This is the one thing the code can't do for you — it needs
  a working key and patience with the free-tier rate limit. The disk cache
  resumes an interrupted run for free, and the writer is now independent of the
  targets, so a scaled run is scientifically clean.
- **Self-perturbation confound — addressed.** Each cell records `self_perturbed`,
  and `--writer-model <not-a-target>` removes it. Re-run the headline on the
  independent-writer subset (`score.writer_effect`).
- **Mechanistic ↔ behavioural — addressed.** `mech --validate` runs both on the
  *same* local model; the cross-model comparison is still reported but marked as
  the weaker one.
- **Labels are automatic.** κ against human/judge is how we measure how far to
  trust them; the annotation UI exists to collect those labels.
- **gpt-oss-120b has a hidden reasoning channel** that can be lowered but not
  switched off, so for that model the perturbed chain is not its only reasoning.

References the project builds on: Lanham et al. 2023 (*Measuring Faithfulness in
CoT*), Turpin et al. 2023 (*LMs Don't Always Say What They Think*), Sprague et
al. 2024 (*To CoT or not to CoT?*). Verify exact citations before use.
