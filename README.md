# Stepwise

**Is a model's chain of thought load-bearing, or is it a story told after the
fact?** Stepwise answers that with five step-level interventions — four that
corrupt a step and one that deletes it — a **pair probe** that moves two steps
at once, a four-test battery over whole chains, a mechanistic cross-check,
validated measurement, and an agentic analysis layer, over open models on the
Groq free tier.

A chain of thought is only an explanation if the answer depends on it. Stepwise
tests that dependence directly: it corrupts a model's *own* reasoning, re-runs
the model, and measures what the answer does — behaviourally (does the answer
move?) and mechanistically (does the probability of the answer fall?).

```bash
pip install -r requirements.txt
cp .env.example .env          # add a free key from https://console.groq.com/keys

python main.py demo                                 # offline, no key - fills every panel
python main.py --dataset gsm8k --n_samples 50       # step-corruption run
python main.py benchmark                            # validate the faithfulness benchmark + manifest
python main.py --dataset probe                      # run a model against the benchmark (KNOWN critical steps)
python main.py --dataset gsm8k --severities minimal,moderate,severe   # dose-response ladder
python main.py --dataset gsm8k --writer-model allam-2-7b   # independent perturbation writer (de-confound)
python main.py battery --dataset gsm8k              # the 5-test battery
python main.py --dataset gsm8k --perturbations fact_substitution,logical_error,arithmetic_change,irrelevant_noise,step_deletion
                                                    # + the ablation arm: delete a step instead of corrupting it
python main.py power --dataset gsm8k                # how many questions the headline actually needs
python main.py interactions --dataset gsm8k         # pair probe: redundancy a single step can't see
python main.py mech --dataset gsm8k --validate --patch     # mechanistic track: same-model check + causal tracing
python main.py introspect --dataset gsm8k          # self-knowledge probe: does the model notice?
python main.py analyse --dataset gsm8k              # agent writes a report, then fact-checks it
python main.py validate --dataset gsm8k             # label reliability + regression
python main.py dashboard                            # http://127.0.0.1:8000
python -m pytest tests/ -q                          # 611 tests, no network, no GPU
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

Those three were measured with a *positional* answer-step control — "the final
step". §1a replaces it with a content-based one and reports the gap, so the
tautology claim is now measured rather than assumed. Re-run
`python main.py score --dataset <run>` to get the corrected figure for any
existing results file; it needs no API calls.

Two questions is a working pipeline, not a result. But the instrument now
separates real reasoning-dependence from the trivial case that the original
headline number hid.

---

## What's in it

### 0. The offline demo — every panel, no key
`python main.py demo` runs the whole instrument against four simulated models
and writes a complete set of results: the corruption grid, the ablation arm,
the pair probe, the four-test battery, the self-knowledge probe, and a run
against the in-repo benchmark. The dashboard then has data in every panel
without an API key, a network connection or a GPU.

The simulated models are not a shortcut past the science — they are specified
by named behavioural parameters (`early_commit`, `filler_survival`,
`bias_susceptibility`, `detection`, `false_alarm`, `necessity`, `redundancy`),
so each panel shows a *differentiated* grid you can read against what each
model was built to be. `sim-detached` answers from zero steps and barely needs
its chain; `sim-faithful` needs all of it; `sim-jumpy` reacts to surface
change, so it scores high sensitivity *and* high control on the benchmark and
therefore near-zero discrimination.

Every one of those runs is written with `simulated: true` and the dashboard
banners it, so a demo number can never be mistaken for a measurement.

### 1. Step corruption — the original probe
Corrupt exactly one step (fact / logic / arithmetic / irrelevant-noise
control), re-run, label **follows / ignores / corrects**. The faithfulness
pipeline: `pipeline.py`, `perturbations.py`, `extractor.py` (lossless
segmentation), `answers.py`.

### 1a. Is the step *stating* the answer? (`leakage.py`)
The tautology above is the project's load-bearing caveat, so how it is detected
matters. It used to be positional — `is_answer_step = step_idx >= n_steps`, "the
last step" — and that proxy is wrong in both directions:

- **Missed leaks.** A model that computes the total at step 3 and spends steps
  4-6 checking it leaked the answer at step 3. Corrupting step 3 is exactly as
  tautological as corrupting the last step, but position calls it reasoning and
  it lands in the headline.
- **False leaks.** A chain ending "Therefore we compare the two totals" states
  no answer at all. Position discarded that cell as tautology; it was evidence.
- **A whole control class thrown away.** An `irrelevant_noise` edit to the
  answer-stating step *leaves the answer in place*. Not tautological — it is a
  genuine control cell, and the positional filter was dropping every one.

`leakage.py` decides it by content, with the extractor's own notion of
"contains an answer" per answer type (a bare capital `B` is an initial, not
option B; short spans get a length floor). Two flags per cell:

- **`states_answer`** — the step displayed the baseline answer, wherever it sat;
- **`answer_overwritten`** — the corruption *took that answer away*, so the
  perturbed trace now shows a different one for the re-run to copy. This is the
  precise tautology, and it is what every "reasoning steps only" figure now
  filters on.

`score.tautology_report` reports dependency four ways over the same cells — all
steps, the positional filter, the content filter, and tautological cells alone
— plus `leaks_outside_final_step` (tautologies the proxy counted as reasoning)
and `final_steps_without_leak` (evidence it threw away). The gap between the
filters is itself a result rather than an assertion.

**It costs nothing to adopt.** `original_step`, `perturbed_step`,
`original_answer` and `answer_type` are already in every record, so this
recomputes over runs finished months ago with no API calls. Records that predate
the fields fall back to the positional proxy **per row**, so every number a
previous run reported is unchanged until it is re-run.

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

### 5a. The Stepwise Faithfulness Benchmark — ground truth (`data/`, `benchmark.py`, `probe_score.py`)
Everywhere else "load-bearing" is *inferred*: corrupt a step, see if the answer
moves, report a rate — but there's no ground truth to check the rate against.
This benchmark supplies it. It's a hand-authored, in-repo set (**50 items**
across all domains, in three difficulty tiers) where each item's **critical**
step (the answer genuinely depends on it), **redundant** steps (restatements,
meta-comments), and — on adversarial items — **distractor** steps (dressed up to
look pivotal but inert) are annotated. Every model reasons over the *same*
authored chain (`--use-reference-cot`, auto-on for `--dataset probe`), so a
corruption has a *correct* outcome:

- corrupt the **critical** step → a faithful model's answer **should move**;
- corrupt a **distractor**, **redundant**, or the **noise control** → it **should not**.

`probe_score` reports **sensitivity**, **distractor follow**, **control
follow**, and **discrimination = sensitivity − control** — the single number for
how much a model's stated reasoning is actually doing the work. The benchmark
apparatus (`benchmark.py`) adds structural **validation**, a frozen,
checksummed **manifest** (`python main.py benchmark`), tiered scoring, and an
independent **annotator check** (`--llm-check`) that has a second model pick
each item's load-bearing step and reports agreement. See
[docs/BENCHMARK.md](docs/BENCHMARK.md).

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

### 5d. The ablation arm — necessity, not error propagation (`step_deletion`)
Every corruption class injects an *error*, which confounds two things the label
cannot tell apart: the reasoning was load-bearing, and the model *noticed* the
error and routed around it. `introspect.py` shows models sometimes do notice.

Deleting a step injects nothing to notice. What is left is the cleaner causal
question — was this step **needed**? The step is removed and the surviving
markers are **renumbered**, because a chain reading "Step 1, Step 3" is a
visible tell that something was taken out.

```bash
python main.py --dataset gsm8k --perturbations fact_substitution,logical_error,arithmetic_change,irrelevant_noise,step_deletion
```

It is deliberately **not** one of the four default classes: default runs cost
and contain exactly what they always did, the mechanistic track (which iterates
those classes) is untouched, and — most importantly — `follows` means something
different here, so pooling the two arms would make `dependency` a mixture of
two claims. `score.necessity_report` reads it instead, and the cross-tab
against the corruption cells *on the same steps* is the payoff:

| cross-tab cell | what it means |
|---|---|
| needed, but did **not** follow an error in it | the model depends on the step without blindly trusting it — the best case, and invisible to corruption alone |
| **not** needed, yet the answer followed | the answer tracked a step the conclusion does not rest on: surface reactivity |

A deletion spends no API call and writes no cache entry — it is a structural
edit, not a generated one — so the arm costs one re-run per cell and nothing
else. It carries no writer, so it has no self-perturbation confound to report.

### 5f. Two steps at once — the blind spot in one-at-a-time (`interactions.py`)
Every other intervention here moves exactly one step, and that design has a
blind spot built into it.

Suppose a chain reaches its answer by **two routes** — it computes the total,
then derives it again a different way. Corrupt either step alone and the
answer does not move, because the other route still carries it. The
single-step probe records `ignores` twice and concludes the reasoning was not
load-bearing. The opposite is true: it was load-bearing *twice over*.
**Redundancy is indistinguishable from unfaithfulness under single-step
intervention**, and no amount of extra single-step cells fixes that — it is a
property of the design, not the sample size.

Corrupting both at once is the only intervention that can separate them:

```bash
python main.py interactions --dataset gsm8k          # targeted by default
```

Each pair is placed against its two single-step outcomes:

| class | meaning |
|---|---|
| **synergistic** | neither step moved the answer alone, together they did — **a backup path the single-step arm scored as inert reasoning** |
| additive | at least one step already moved it alone; the pair inherits that |
| masked | a step moved it alone but the pair does not — the model noticing the trace has become incoherent and re-deriving |
| inert | nothing moved, alone or together |

`synergy_rate` is the headline: of the pairs where *neither* step moved the
answer alone, how often moving both did. That is the fraction of
apparently-inert reasoning that was actually redundant, and nothing else in
the project can produce it.

**Keeping it affordable.** Pairs are quadratic — eight steps give 28 per class,
more calls than the whole single-step grid. Two things hold that down. The grid
is subsampled deterministically (`--max-pairs`), and by default the probe is
**targeted**: it reads the existing run and draws pairs only from steps whose
single-step cells *both* came back `ignores`, because that is exactly where a
backup path can hide. A pair containing a step that already moves the answer
alone can only ever come out `additive`, so targeting makes the probe cheaper
*and* sharper than a random sweep. Each half reuses the single-step arm's
cached perturbation, so the edits cost nothing twice.

Pair records go to their own sidecar (`<run>.interactions.jsonl`) and are never
mixed into the main results, so every existing score and table is unchanged by
running this.

### 5e. How big does the run need to be? (`stats.py`, `main.py power`)
The project's standing limitation is scale: the headline needs 50+ questions and
the committed numbers come from 2. "How many do I need" has a real answer, and
the naive one is badly wrong here — the unit of sampling is the **question**,
not the cell, and one question's cells share a chain of thought.

```bash
python main.py power --dataset gsm8k --effect 0.40,0.25   # or the Diagnostics panel
```

Reads cells-per-question and the **intra-cluster correlation** off the run
itself (`stats.observed_icc`), then reports the **design effect**
`1 + (m-1)·icc` and inflates the two-proportion sample size by it. At 8 cells a
question that is 2.1x at an ICC of 0.15 and 4.2x at 0.45, so ignoring it can
under-size a run several-fold. Also the inverse: `stats.precision_for` gives
the 95% CI half-width a run of a given size can actually support — at 8 cells a
question and an ICC of 0.3, two questions buy a half-width of about
**±42 points** and fifty buy about **±8**. Run the command on your own results
to get the figure from *your* measured clustering rather than those defaults;
that number is the honest version of "this is a working pipeline, not a result".

### 6. Agents + adversarial verifier (`agent/`)
- A **report agent** and a streaming **chat agent**, each a real tool-calling
  loop over fourteen query tools — they decide what to compute, and every number
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
The findings surface — the tautology gap (positional proxy against the
content-based test), the ablation arm's necessity cross-tab, a sizing panel
that plots the CI half-width against run size with this run and the target
marked, domain comparison with CIs, per-step grid, the side-by-side explorer
with inline human-annotation buttons, the agent report, and the streaming Ask
panel. FastAPI serves the API and the page as one process.

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
pytest tests/ -q      # 611 tests, all offline (scripted provider, fake scorer)
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
  score.py           aggregation, chance correction, tautology split, CIs
  leakage.py         does a step STATE the answer, and did the edit remove it
  stats.py           chance rate, clustered bootstrap, design effect, sizing
  domains.py         reasoning domains above datasets
  interactions.py    pair interventions: synergy, additivity, masking
  battery.py         early answering, paraphrase, filler, biased context
  battery_score.py   one metric per battery test
  mech.py            log-prob step effect, same-model probe, activation patching
  introspect.py      self-knowledge probe (does the model notice?)
  benchmark.py       the faithfulness benchmark: validation, manifest, tiered report
  probe_score.py     sensitivity/distractor/control against known critical steps
  validation.py      Cohen's κ, annotation store, regression
  judge.py           independent LLM judge
  datasets.py        nine loaders + fixtures (incl. the in-repo probe set)
  demo.py            offline simulated run
  mcp_server.py      tools over MCP (stdio)
  agent/
    tools.py         fourteen query tools
    analyst.py       report agent (+ auto verification)
    chat.py          streaming chat agent
    verifier.py      adversarial fact-checker
  dashboard/
    app.py           FastAPI API + SSE chat + annotation endpoints
    static/index.html the page
tests/               611 tests, no network
docs/                call map, deploy guide, MCP config
```

## Limitations

- **Scale — now quantified.** The real finding needs 50+ questions across
  domains; the current numbers are from 2. This is the one thing the code can't
  do for you — it needs a working key and patience with the free-tier rate
  limit. `python main.py power` turns the shrug into a number: it reads the
  clustering off whatever run exists and says how many questions the target
  effect needs, what precision the current one supports, and roughly what the
  calls cost. The disk cache resumes an interrupted run for free, and the
  writer is now independent of the targets, so a scaled run is clean.
- **Answer-step tautology — now measured, not assumed.** The control was
  positional ("the last step"), which both over- and under-fires. §1a replaces
  it with the content-based `answer_overwritten` test and reports the gap. Old
  records fall back to the positional proxy per row, so nothing already
  reported silently changed.
- **Error-detection confound — addressed by the ablation arm.** A corruption
  measures error propagation, which a model can defeat by *noticing*. §5d
  deletes the step instead, so "necessary" is measured with no error to spot.
- **Single-step blindness to redundancy — addressed by the pair arm.** One
  intervention at a time cannot tell inert reasoning from reasoning done twice;
  both read as `ignores`. §5f moves two steps at once and reports the synergy
  rate. Note the converse limit: a chain with *three* routes needs triples, and
  the grid is cubic, so the pair probe bounds the problem rather than closing
  it — a zero synergy rate is evidence against two-way redundancy only.
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
