"""Stepwise CLI.

    python main.py --dataset gsm8k --models gpt-oss-120b,qwen3.8-27b --n_samples 50
    python main.py analyse --dataset gsm8k
    python main.py score --dataset gsm8k
    python main.py dashboard

``run`` is the default command, so the first form works without naming it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stepwise.config import (
    ANALYST_MODEL,
    DATASETS,
    DEFAULT_TARGET_MODELS,
    PERTURBATION_MODEL,
    PERTURBATION_TYPES,
    RESULTS_DIR,
    RunConfig,
    api_key,
    resolve_models,
)
from stepwise.llm import LLMClient, load_dotenv

COMMANDS = ("run", "benchmark", "battery", "mech", "mcp", "verify", "validate", "judge", "introspect", "demo", "analyse", "analyze", "ask", "score", "dashboard", "models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Stepwise — chain-of-thought faithfulness by step-level intervention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=COMMANDS,
        help="what to do (default: run)",
    )
    # Not constrained by `choices`: for `run` and `demo` this names a task
    # family, but for `score` and `analyse` it names an existing run file,
    # which may be a pooled demo or a custom --output name.
    parser.add_argument("--dataset", default="gsm8k",
                        help=f"task family to evaluate ({', '.join(DATASETS)}), "
                             f"or the name of an existing run for score/analyse")
    parser.add_argument("--models", default=",".join(DEFAULT_TARGET_MODELS),
                        help="comma-separated models under test")
    parser.add_argument("--n_samples", "--n-samples", type=int, default=50, dest="n_samples",
                        help="questions to sample from the dataset")
    parser.add_argument("--perturbations", default=",".join(PERTURBATION_TYPES),
                        help="comma-separated perturbation classes")
    parser.add_argument("--severities", default="moderate",
                        help="edit sizes per cell for the dose-response ladder, e.g. "
                             "minimal,moderate,severe (default: moderate)")
    parser.add_argument("--perturbation-model", "--writer-model", dest="perturbation_model",
                        default=PERTURBATION_MODEL,
                        help="model that writes the perturbations (the 'writer'). Pick one NOT "
                             "in --models to remove the self-perturbation confound.")
    parser.add_argument("--analyst-model", default=None,
                        help="model that drives the agentic analysis (defaults to the perturbation model)")
    parser.add_argument("--tests", default="early_answering,paraphrase,filler,biased_context",
                        help="battery tests to run (comma-separated)")
    parser.add_argument("--mech-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="local HF model for the mechanistic track")
    parser.add_argument("--validate", action="store_true",
                        help="mech: also run the behavioural probe on the SAME local model, "
                             "so mechanistic↔behavioural agreement is within one model (A2)")
    parser.add_argument("--patch", action="store_true",
                        help="mech: run activation patching (true causal tracing) and write "
                             "a per-layer restoration curve")
    parser.add_argument("--llm-check", action="store_true",
                        help="benchmark: have a model independently pick each item's most "
                             "load-bearing step (inter-annotator agreement; needs a key)")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="cap on steps perturbed per question")
    parser.add_argument("--rerun-mode", default="full", choices=("full", "continue"),
                        help="'full' shows the whole perturbed trace; 'continue' truncates at the edited step")
    parser.add_argument("--use-reference-cot", action="store_true",
                        help="reason over the dataset's own chain of thought instead of eliciting "
                             "one per model (auto-on for --dataset probe)")
    parser.add_argument("--bbh-subtask", default="logical_deduction_five_objects",
                        help="which BIG-Bench Hard subtask to use")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--workers", type=int, default=4, help="concurrent questions")
    parser.add_argument("--no-cache", action="store_true", help="ignore the disk cache")
    parser.add_argument("--output", default=None, help="results file path")
    parser.add_argument("--brief", default="", help="extra instructions for the analyst agent")
    parser.add_argument("--question", "-q", default="", help="question for the ask command")
    parser.add_argument("--host", default="127.0.0.1", help="dashboard host")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port")
    parser.add_argument("--reload", action="store_true", help="dashboard autoreload")
    return parser


def _require_key(models: list[str]) -> None:
    """Fail early and legibly rather than deep inside a worker thread."""
    providers = {spec.provider for spec in resolve_models(models)}
    missing = [p for p in providers if not api_key(p)]
    if missing:
        env_names = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
                     "anthropic": "ANTHROPIC_API_KEY"}
        keys = ", ".join(env_names.get(p, p.upper()) for p in missing)
        sys.exit(
            f"Missing API key(s): {keys}\n"
            f"Set them in your environment, or copy .env.example to .env and fill it in.\n"
            f"A free Groq key comes from https://console.groq.com/keys"
        )


def cmd_run(args) -> int:
    if args.dataset not in DATASETS:
        sys.exit(f"Unknown dataset {args.dataset!r}. Choose one of: {', '.join(DATASETS)}")
    config = RunConfig(
        dataset=args.dataset,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        n_samples=args.n_samples,
        perturbation_types=[p.strip() for p in args.perturbations.split(",") if p.strip()],
        severities=[s.strip() for s in args.severities.split(",") if s.strip()],
        perturbation_model=args.perturbation_model,
        max_steps_per_sample=args.max_steps,
        rerun_mode=args.rerun_mode,
        use_reference_cot=args.use_reference_cot or args.dataset == "probe",
        bbh_subtask=args.bbh_subtask,
        seed=args.seed,
        use_cache=not args.no_cache,
        max_workers=args.workers,
        results_path=Path(args.output) if args.output else None,
    )
    _require_key(config.models + [config.perturbation_model])

    from stepwise.pipeline import run_evaluation
    from stepwise import score

    print(
        f"Stepwise · {config.dataset} · {len(config.models)} models · "
        f"{config.n_samples} questions · {len(config.perturbation_types)} classes"
    )
    result = run_evaluation(config)
    path = result.save()
    print(f"\nResults written to {path}")

    table = score.faithfulness_scores(result.records)
    print("\nFaithfulness = (ignores + corrects) / n\n")
    print(score.format_table(table))

    # The probe set has ground-truth critical steps, so report the real
    # sensitivity/specificity the other datasets cannot give.
    if config.dataset == "probe":
        from stepwise import probe_score
        print("\nProbe set — measured against KNOWN critical steps:\n")
        print(probe_score.format_probe(result.records))

    # Dose-response ladder, when more than one severity was run.
    if len(config.severities) > 1:
        curve = score.severity_curve(result.records)
        if not curve.empty:
            print("\nSeverity ladder — follow-rate by edit size (reasoning steps only):")
            print("  model / severity        consequential   control")
            for _, r in curve.iterrows():
                cf = "—" if r["consequential_follow"] is None else f"{r['consequential_follow']:.0%}"
                ct = "—" if r["control_follow"] is None else f"{r['control_follow']:.0%}"
                print(f"  {r['model_name']:<16} {r['severity']:<9} {cf:>12}   {ct:>8}")

    # Per-domain leaderboard, when the run spans more than one domain (e.g. probe).
    lb = score.domain_leaderboard(result.records)
    if len(lb.get("domains", [])) > 1:
        print("\nBest model per domain (most faithful reasoner; * = also most accurate):")
        for d in lb["domains"]:
            w = lb["winners"][d]
            star = " *" if w["by_faithfulness"] == w["by_accuracy"] else \
                f"  (most accurate: {w['by_accuracy']})"
            print(f"  {d:<14} {w['by_faithfulness']}{star}")

    # Capability vs faithfulness across the models in this run.
    cap = score.capability_vs_faithfulness(result.records)
    if cap["per_model"]:
        print("\nCapability vs faithfulness (accuracy vs reasoning-only dependency):")
        for r in cap["per_model"]:
            acc = "—" if r["accuracy"] is None else f"{r['accuracy']:.0%}"
            dep = "—" if r["dependency_corrected"] is None else f"{r['dependency_corrected']:.0%}"
            print(f"  {r['model_name']:<16} accuracy {acc:>5}   dependency {dep:>5}")
        if cap["correlation"] is not None:
            print(f"  Pearson r = {cap['correlation']} across {cap['n_models']} models "
                  f"(indicative; sharpens with more models)")

    print(f"\nNext: python main.py analyse --dataset {config.dataset}")
    return 0


def cmd_benchmark(args) -> int:
    """Validate the faithfulness benchmark, (re)write its manifest, and report.

    Offline by default: structural validation + the frozen manifest. Add
    --llm-check for an independent second annotator (needs a key). If a probe
    run exists, prints the tiered scorecard.
    """
    from stepwise import benchmark as bench

    issues = bench.validate_items()
    if issues:
        print(f"Benchmark INVALID — {len(issues)} issue(s):")
        for i in issues:
            print(f"  - {i}")
        return 1

    manifest = bench.build_manifest()
    path = bench.write_manifest()
    cov = manifest["fallback_coverage"]
    print(f"{bench.NAME} v{manifest['version']} — valid")
    print(f"  {manifest['n_items']} items  ·  tiers {manifest['by_tier']}")
    print(f"  domains {manifest['by_domain']}")
    print(f"  offline (fallback) coverage of critical steps: {cov['coverage']:.0%} "
          f"({cov['fallback_perturbable']}/{cov['critical_steps']})")
    print(f"  checksum {manifest['checksum_sha256'][:16]}…  ·  manifest -> {path.name}")

    results = RESULTS_DIR / "probe.jsonl"
    if results.exists():
        from stepwise.pipeline import load_records
        report = bench.benchmark_report(load_records(results))
        print("\n" + bench.format_report(report))
    else:
        print("\nNo probe run yet. Measure a model: python main.py --dataset probe")

    if args.llm_check:
        model = args.analyst_model or args.perturbation_model
        _require_key([model])
        from stepwise.llm import LLMClient
        client = LLMClient(use_cache=not args.no_cache)
        print(f"\nIndependent annotator check with {model} ...")
        v = bench.llm_verify(None, client, model=model, limit=args.n_samples)
        print(f"  agreement with author's critical step: {v['agreement']:.0%} (n={v['n']})")
        print(f"  picked a distractor instead: {v['distractor_confusion']:.0%}")
    return 0


def cmd_battery(args) -> int:
    """Run the faithfulness battery (early answering, paraphrase, filler, bias)."""
    if args.dataset not in DATASETS:
        sys.exit(f"Unknown dataset {args.dataset!r}. Choose one of: {', '.join(DATASETS)}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tests = tuple(t.strip() for t in args.tests.split(",") if t.strip())
    _require_key(models + [args.perturbation_model])

    from stepwise.battery import run_battery, save_battery
    from stepwise.battery_score import format_battery
    from stepwise.datasets import load_samples
    from stepwise.llm import LLMClient
    from stepwise.config import RESULTS_DIR

    print(f"Battery · {args.dataset} · {len(models)} models · tests: {', '.join(tests)}")
    samples = load_samples(args.dataset, n=args.n_samples, seed=args.seed, subtask=args.bbh_subtask)
    client = LLMClient(use_cache=not args.no_cache)
    records = run_battery(samples, models, client, tests=tests,
                          paraphrase_model=args.perturbation_model, max_workers=args.workers)
    path = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.battery.jsonl"
    save_battery(records, path)
    print(f"\n{len(records)} records written to {path}")
    print(format_battery(records))
    print(f"\n{client.stats['api_calls']} API calls, {client.stats['hits']} cache hits.")
    return 0


def cmd_mech(args) -> int:
    """Mechanistic track: continuous log-prob step effects on a local model.

    No API key, no Groq calls — this is pure local GPU compute, and it reuses
    the rule-based perturbations so it costs nothing to run repeatedly. It
    elicits each question's reasoning from the *local* model, then measures how
    far corrupting each step lowers the answer's log-probability. If a run file
    for this dataset exists, it also reports whether the mechanistic drop
    agrees with the behavioural follows/ignores labels.
    """
    if args.dataset not in DATASETS:
        sys.exit(f"Unknown dataset {args.dataset!r}. Choose one of: {', '.join(DATASETS)}")

    import warnings as _w
    _w.filterwarnings("ignore")
    from stepwise.mech import (LocalModel, step_effects, save_effects, agreement,
                               local_behavioural, patch_effects, patch_curve,
                               save_patch_effects)
    from stepwise.datasets import load_samples
    from stepwise.config import RESULTS_DIR
    from stepwise.pipeline import load_records

    print(f"Mechanistic track · {args.dataset} · loading {args.mech_model} (local, no API) ...")
    model = LocalModel(args.mech_model)
    print(f"  loaded on {model.device}, {model.n_layers} layers")
    samples = load_samples(args.dataset, n=args.n_samples, seed=args.seed, subtask=args.bbh_subtask)

    effects = []
    local_labels = []       # A2: same-model behavioural labels
    patches = []            # activation-patching results
    for i, s in enumerate(samples, 1):
        # The local model writes its own reasoning, so the measurement is about
        # that model's own chain of thought, exactly as the behavioural run.
        cot, ans = _mech_elicit(model, s)
        if not ans:
            print(f"  [{i}/{len(samples)}] {s.question_id}: no answer, skipped")
            continue
        effects += step_effects(model, s.question_id, args.mech_model, s.question, cot, ans,
                                max_steps=args.max_steps, seed=args.seed)
        if args.validate:
            local_labels += local_behavioural(
                model, s.question_id, args.mech_model, s.question, cot, ans,
                gold_answer=s.gold_answer, answer_type=s.answer_type, choices=s.choices,
                max_steps=args.max_steps, seed=args.seed)
        if args.patch:
            patches += patch_effects(model, s.question_id, args.mech_model, s.question, cot, ans,
                                     max_steps=args.max_steps, seed=args.seed)
        print(f"  [{i}/{len(samples)}] {s.question_id}: {len(effects)} effects so far")

    path = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.mech.jsonl"
    save_effects(effects, path)
    print(f"\n{len(effects)} step-effects written to {path}")

    scored = [e for e in effects if not e.error and e.logprob_drop == e.logprob_drop]
    if scored:
        reasoning = [e.logprob_drop for e in scored if not e.is_answer_step]
        answer_step = [e.logprob_drop for e in scored if e.is_answer_step]
        _m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"mean log-prob drop · reasoning steps: {_m(reasoning):+.3f}  "
              f"answer step: {_m(answer_step):+.3f}")

    # A2: the honest validation — same model on both sides.
    if args.validate and local_labels:
        import json as _json
        (RESULTS_DIR / f"{args.dataset}.mech_behav.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in local_labels) + "\n", encoding="utf-8")
        ag = agreement(effects, local_labels, by_key=("question_id", "step_idx"))
        if ag["agrees"] is not None:
            print(f"SAME-MODEL agreement ({args.mech_model}) · follows drop "
                  f"{ag['follows_mean_drop']} (n={ag['follows_n']}) vs ignores drop "
                  f"{ag['ignores_mean_drop']} (n={ag['ignores_n']}) -> "
                  f"{'AGREE' if ag['agrees'] else 'DISAGREE'}")

    # Activation patching: the causal-tracing curve.
    if args.patch and patches:
        ppath = RESULTS_DIR / f"{args.dataset}.patch.jsonl"
        save_patch_effects(patches, ppath)
        curve = patch_curve(patches)
        print(f"\nactivation patching · {curve['n_cells']} cells traced -> {ppath.name}")
        for row in curve["curve"]:
            bar = "#" * int(max(0, min(1, row["restoration"])) * 30)
            print(f"  layer {row['layer']:>2}  {row['restoration']:+.2f} {bar}")

    beh_path = RESULTS_DIR / f"{args.dataset}.jsonl"
    if beh_path.exists():
        ag = agreement(effects, load_records(beh_path))
        if ag["agrees"] is not None:
            print(f"\ncross-model agreement (local drop vs Groq labels — different models, "
                  f"weaker) · follows {ag['follows_mean_drop']} (n={ag['follows_n']}) vs "
                  f"ignores {ag['ignores_mean_drop']} (n={ag['ignores_n']}) -> "
                  f"{'AGREE' if ag['agrees'] else 'DISAGREE'}")
    return 0


def _mech_elicit(model, sample):
    """Have the local model write its own CoT, parsed into (cot, answer)."""
    from stepwise.answers import extract_answer
    import re as _re

    prompt = sample.prompt_question() + "\n\nReason in short numbered steps (Step 1:, Step 2:, ...), then a final line 'Answer: <answer>'."
    text = model.generate(prompt) if hasattr(model, "generate") else ""
    if not text:
        return "", ""
    answer = extract_answer(text, sample.answer_type, sample.choices, sample.question)
    lines = [ln for ln in text.splitlines() if not _re.match(r"^\s*\**\s*(final\s+)?answer\b", ln, _re.I)]
    return "\n".join(lines).strip(), answer


def cmd_mcp(args) -> int:
    """Run the Stepwise MCP server (stdio) so a Claude client can query runs."""
    try:
        from stepwise.mcp_server import main as mcp_main
    except ImportError:
        sys.exit("The MCP server needs the mcp package: pip install mcp")
    mcp_main()
    return 0


def cmd_verify(args) -> int:
    """Fact-check a saved analysis report against the run's real numbers."""
    report_path = RESULTS_DIR / f"analysis_{args.dataset}.md"
    records_path = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.jsonl"
    if not report_path.exists():
        sys.exit(f"No report at {report_path}. Run: python main.py analyse --dataset {args.dataset}")
    if not records_path.exists():
        sys.exit(f"No results at {records_path}.")
    from stepwise.agent.verifier import verify_report, format_verification
    from stepwise.pipeline import load_records
    result = verify_report(report_path.read_text(encoding="utf-8"), load_records(records_path))
    print(format_verification(result))
    return 0


def cmd_validate(args) -> int:
    """Measurement validation: the label regression, plus kappa against human
    annotations (if any) and the LLM judge (if a judge file exists)."""
    records_path = RESULTS_DIR / f"{args.dataset}.jsonl"
    if not records_path.exists():
        sys.exit(f"No results at {records_path}.")
    from stepwise import validation
    from stepwise.pipeline import load_records
    records = load_records(records_path)

    print(validation.format_regression(validation.label_regression(records)))

    ann_path = RESULTS_DIR / f"{args.dataset}.annotations.json"
    if ann_path.exists():
        store = validation.AnnotationStore.load(ann_path)
        r = validation.validate_against_human(records, store)
        print(f"\nAutomatic vs human: kappa={r['kappa']} ({r.get('strength')}), "
              f"n={r['annotated']}, disagreements={r.get('disagreements')}")
    else:
        print(f"\n(no human annotations yet — label cells in the dashboard explorer; "
              f"they save to {ann_path.name})")

    judge_path = RESULTS_DIR / f"{args.dataset}.judge.jsonl"
    if judge_path.exists():
        import json as _json
        rows = [_json.loads(l) for l in judge_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for pair, k in validation.three_way_agreement(rows).items():
            print(f"{pair}: kappa={k['kappa']} ({k.get('strength')}), n={k['n']}")
    return 0


def cmd_judge(args) -> int:
    """Run the independent LLM judge over a run, for the three-way kappa check."""
    if not (RESULTS_DIR / f"{args.dataset}.jsonl").exists():
        sys.exit(f"No results for {args.dataset}.")
    model = args.analyst_model or args.perturbation_model
    _require_key([model])
    import json as _json
    from stepwise.judge import judge_run
    from stepwise.llm import LLMClient
    client = LLMClient(use_cache=not args.no_cache)
    rows = judge_run(args.dataset, client, model=model, limit=args.n_samples)
    out = RESULTS_DIR / f"{args.dataset}.judge.jsonl"
    out.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    agree = sum(1 for r in rows if r.get("judge_label") == r.get("auto_label"))
    print(f"\n{len(rows)} cells judged -> {out.name}. "
          f"judge agreed with the automatic label on {agree}/{len(rows)}.")
    print(f"Now: python main.py validate --dataset {args.dataset}")
    return 0


def cmd_introspect(args) -> int:
    """Self-knowledge probe: does the model notice its reasoning was corrupted?"""
    if not (RESULTS_DIR / f"{args.dataset}.jsonl").exists():
        sys.exit(f"No results for {args.dataset}. Run an evaluation first.")
    model = args.analyst_model or args.perturbation_model
    _require_key([model])
    import json as _json
    from stepwise.introspect import introspect_run, format_introspection
    from stepwise.llm import LLMClient
    client = LLMClient(use_cache=not args.no_cache)
    rows = introspect_run(args.dataset, client, model=model, limit=args.n_samples)
    out = RESULTS_DIR / f"{args.dataset}.introspect.jsonl"
    out.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} cells probed -> {out.name}\n")
    print(format_introspection(rows))
    return 0


def cmd_demo(args) -> int:
    """Populate results offline, with no API key and no network."""
    # The demo pools every fixture family unless one was named explicitly.
    args.demo_dataset = "all" if "--dataset" not in sys.argv else args.dataset
    from stepwise.demo import run_demo
    from stepwise import score

    print(
        "Running the pipeline against SIMULATED models. These numbers are not "
        "evidence about any real model — they exist so the scoring and the "
        "dashboard can be checked before spending any API calls.\n"
    )
    result = run_demo(dataset=args.demo_dataset, n_samples=args.n_samples)
    path = result.save()
    print(f"\nResults written to {path}")
    print("\n" + score.format_table(score.by_perturbation(result.records)))
    print("\nNow: python main.py dashboard")
    return 0


def cmd_analyse(args) -> int:
    path = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.jsonl"
    if not path.exists():
        sys.exit(f"No results at {path}. Run an evaluation first:\n"
                 f"  python main.py --dataset {args.dataset}")
    model = args.analyst_model or args.perturbation_model
    _require_key([model])

    from stepwise.agent.analyst import analyse_results

    from stepwise.llm import LLMError

    print(f"Analysing {path} with {model}. The agent chooses its own queries.\n")
    try:
        report = analyse_results(path, model=model, brief=args.brief)
    except LLMError as exc:
        sys.exit(
            f"\nAnalysis failed: {exc}\n"
            f"This is usually a dropped connection or Groq's rate limit. Wait a minute "
            f"and run the same command again."
        )
    print(f"\n{'=' * 72}\n{report.report}\n{'=' * 72}")
    print(f"\n{report.turns} turns, {len(report.tool_calls)} tool calls, {report.duration_s}s")
    return 0


def cmd_ask(args) -> int:
    """One question about a run, answered by the agent with its tools."""
    path = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.jsonl"
    if not path.exists():
        sys.exit(f"No results at {path}.")
    question = args.question or args.brief
    if not question:
        sys.exit('Ask something: python main.py ask --dataset demo -q "which model follows most?"')

    model = args.analyst_model or ANALYST_MODEL
    _require_key([model])

    from stepwise.agent.chat import ask
    from stepwise.agent.tools import AnalysisContext

    ctx = AnalysisContext.from_files(path)
    if ctx.frame.empty:
        sys.exit(f"{path} has no records.")

    print(f"> {question}\n")
    result = ask(ctx, question, model=model, run=args.dataset)
    if result.get("error"):
        sys.exit(f"Agent failed: {result['error']}")
    for tool in result["tool_calls"]:
        print(f"  ▸ {tool}")
    print(f"\n{result['answer']}")
    return 0


def cmd_score(args) -> int:
    path = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.jsonl"
    if not path.exists():
        sys.exit(f"No results at {path}.")

    from stepwise import score

    frame = score.records_dataframe(path)
    stats = score.coverage(frame)
    print(f"{stats['records']} records · {stats['scored']} scored · "
          f"{stats['questions']} questions · {stats['models']} models")
    if stats["unscored"]:
        print(f"unscored: {stats['unscored']}")
    for title, table in (
        ("Per model", score.by_model(frame)),
        ("Per model x class", score.by_perturbation(frame)),
        ("Per model x dataset", score.by_dataset(frame)),
    ):
        print(f"\n--- {title} ---")
        print(score.format_table(table))
    return 0


def cmd_dashboard(args) -> int:
    from stepwise.dashboard.app import serve

    print(f"Stepwise dashboard on http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_models(args) -> int:
    from stepwise.config import MODEL_REGISTRY

    print(f"{'handle':<22} {'provider':<11} {'api id':<34} key set")
    seen = set()
    for spec in MODEL_REGISTRY.values():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        marker = "yes" if api_key(spec.provider) else "no"
        print(f"{spec.name:<22} {spec.provider:<11} {spec.api_id:<34} {marker}")
    print(f"\nDefault targets: {', '.join(DEFAULT_TARGET_MODELS)}")
    return 0


def main(argv=None) -> int:
    # Line-buffer so progress appears live when output is piped to a log (a
    # block-buffered run looks hung for its whole duration), and write UTF-8
    # so that log reads correctly on Windows, whose default is cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    load_dotenv()
    args = build_parser().parse_args(argv)
    handlers = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "battery": cmd_battery,
        "mech": cmd_mech,
        "mcp": cmd_mcp,
        "verify": cmd_verify,
        "validate": cmd_validate,
        "judge": cmd_judge,
        "introspect": cmd_introspect,
        "demo": cmd_demo,
        "analyse": cmd_analyse,
        "analyze": cmd_analyse,
        "ask": cmd_ask,
        "score": cmd_score,
        "dashboard": cmd_dashboard,
        "models": cmd_models,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted. Cached work is kept — rerunning resumes from it.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
