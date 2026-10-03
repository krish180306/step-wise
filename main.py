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

COMMANDS = ("run", "battery", "mech", "mcp", "verify", "demo", "analyse", "analyze", "ask", "score", "dashboard", "models")


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
    parser.add_argument("--perturbation-model", default=PERTURBATION_MODEL,
                        help="model that writes the perturbations")
    parser.add_argument("--analyst-model", default=None,
                        help="model that drives the agentic analysis (defaults to the perturbation model)")
    parser.add_argument("--tests", default="early_answering,paraphrase,filler,biased_context",
                        help="battery tests to run (comma-separated)")
    parser.add_argument("--mech-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="local HF model for the mechanistic track")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="cap on steps perturbed per question")
    parser.add_argument("--rerun-mode", default="full", choices=("full", "continue"),
                        help="'full' shows the whole perturbed trace; 'continue' truncates at the edited step")
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
        perturbation_model=args.perturbation_model,
        max_steps_per_sample=args.max_steps,
        rerun_mode=args.rerun_mode,
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
    print(f"\nNext: python main.py analyse --dataset {config.dataset}")
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
    from stepwise.mech import LocalModel, step_effects, save_effects, agreement
    from stepwise.datasets import load_samples
    from stepwise.config import RESULTS_DIR
    from stepwise.pipeline import load_records

    print(f"Mechanistic track · {args.dataset} · loading {args.mech_model} (local, no API) ...")
    model = LocalModel(args.mech_model)
    print(f"  loaded on {model.device}, {model.n_layers} layers")
    samples = load_samples(args.dataset, n=args.n_samples, seed=args.seed, subtask=args.bbh_subtask)

    effects = []
    for i, s in enumerate(samples, 1):
        # The local model writes its own reasoning, so the measurement is about
        # that model's own chain of thought, exactly as the behavioural run.
        cot, ans = _mech_elicit(model, s)
        if not ans:
            print(f"  [{i}/{len(samples)}] {s.question_id}: no answer, skipped")
            continue
        effects += step_effects(model, s.question_id, args.mech_model, s.question, cot, ans,
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
    beh_path = RESULTS_DIR / f"{args.dataset}.jsonl"
    if beh_path.exists():
        ag = agreement(effects, load_records(beh_path))
        if ag["agrees"] is not None:
            print(f"behavioural agreement · follows drop {ag['follows_mean_drop']} "
                  f"(n={ag['follows_n']}) vs ignores drop {ag['ignores_mean_drop']} "
                  f"(n={ag['ignores_n']}) -> {'AGREE' if ag['agrees'] else 'DISAGREE'}")
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
        "battery": cmd_battery,
        "mech": cmd_mech,
        "mcp": cmd_mcp,
        "verify": cmd_verify,
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
