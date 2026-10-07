"""A one-page model report card: every faithfulness signal for a run, per
model, in a single standalone HTML file you can send to someone.

The dashboard is for exploring; this is for *handing over*. It gathers whatever
the run produced — the step-corruption scores, the battery, the self-knowledge
probe, the per-domain leaderboard, and (for a probe run) the benchmark
discrimination — and lays out one card per model plus the cross-model verdict.
Missing pieces are shown as "not run" rather than omitted, so the card also
tells you what is still uncovered.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

from .config import RESULTS_DIR
from .pipeline import load_records


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_report(dataset: str, results_dir: Path | None = None) -> dict:
    """Collect every available signal for ``dataset`` into one structure."""
    from . import score
    rd = Path(results_dir or RESULTS_DIR)
    main = rd / f"{dataset}.jsonl"
    records = load_records(main)
    summ = score.summary(records) if records else {}

    # Per-model core, keyed by model.
    reasoning = {r["model_name"]: r for r in summ.get("by_model_reasoning", [])}
    allsteps = {r["model_name"]: r for r in summ.get("by_model", [])}
    # Control follow (irrelevant_noise) per model.
    control = {}
    for r in summ.get("by_perturbation", []):
        if r.get("perturbation_type") == "irrelevant_noise":
            control[r["model_name"]] = r.get("dependency")

    # Sidecars.
    battery = {}
    try:
        from .battery_score import battery_summary
        for r in battery_summary(_read_jsonl(rd / f"{dataset}.battery.jsonl")).get("per_model", []):
            battery[r["model_name"]] = r
    except Exception:
        pass
    introspect = {}
    try:
        from .introspect import introspect_summary
        for r in introspect_summary(_read_jsonl(rd / f"{dataset}.introspect.jsonl")):
            introspect[r["model_name"]] = r
    except Exception:
        pass
    probe = {}
    try:
        from .probe_score import probe_scores
        for r in probe_scores(records):
            probe[r["model_name"]] = r
    except Exception:
        pass

    models = sorted(set(reasoning) | set(allsteps) | set(battery) | set(introspect))
    per_model = []
    for m in models:
        rr = reasoning.get(m, {})
        per_model.append({
            "model": m,
            "n": rr.get("n"),
            "accuracy": (allsteps.get(m, {}) or {}).get("accuracy_retained"),
            "faithfulness": rr.get("faithfulness_corrected"),
            "dependency": rr.get("dependency_corrected"),
            "control_follow": control.get(m),
            "battery": battery.get(m),
            "introspect": introspect.get(m),
            "probe": probe.get(m),
        })

    return {
        "dataset": dataset,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "coverage": summ.get("coverage", {}),
        "per_model": per_model,
        "domain_leaderboard": summ.get("domain_leaderboard", {}),
        "is_probe": dataset == "probe" or bool(probe),
        "simulated": any(r.get("simulated") for r in records),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_PCT = lambda v: "—" if v is None else f"{v * 100:.0f}%"
_esc = lambda s: html.escape(str(s))


def _metric(label: str, value: str, hint: str = "") -> str:
    return (f'<div class="metric"><div class="ml">{_esc(label)}</div>'
            f'<div class="mv">{value}</div>'
            f'{f"<div class=mh>{_esc(hint)}</div>" if hint else ""}</div>')


def _card(m: dict) -> str:
    bat = m.get("battery") or {}
    intro = m.get("introspect") or {}
    probe = m.get("probe") or {}
    rows = [
        _metric("accuracy", _PCT(m.get("accuracy")), "clean answers correct"),
        _metric("dependency", _PCT(m.get("dependency")), "reasoning load-bearing"),
        _metric("control follow", _PCT(m.get("control_follow")), "reacts to noise (lower=better)"),
        _metric("faithfulness", _PCT(m.get("faithfulness")), "answer survives a corrupt step"),
    ]
    if bat:
        rows.append(_metric("filler accuracy", _PCT(bat.get("filler_accuracy")), "accuracy on '...' reasoning"))
        rows.append(_metric("early-answer", _PCT(bat.get("early_already_knew")), "knew before reasoning"))
    if intro:
        rows.append(_metric("error detection", _PCT(intro.get("detection_rate")), "notices a planted error"))
        rows.append(_metric("fooled & blind", _PCT(intro.get("fooled_and_blind")), "followed & didn't notice"))
    if probe:
        rows.append(_metric("probe sensitivity", _PCT(probe.get("sensitivity")), "moves when key step breaks"))
        rows.append(_metric("probe discrimination", _PCT(probe.get("discrimination")), "sensitivity − control"))
    n = m.get("n")
    return (f'<section class="card"><h2>{_esc(m["model"])}</h2>'
            f'<div class="n">n = {n if n is not None else "—"} scored cells</div>'
            f'<div class="metrics">{"".join(rows)}</div></section>')


def _leaderboard_html(lb: dict) -> str:
    domains = lb.get("domains") or []
    if not domains:
        return '<p class="muted">Per-domain leaderboard needs more than one domain (run the probe benchmark or several datasets).</p>'
    rows = []
    for d in domains:
        w = (lb.get("winners") or {}).get(d, {})
        margin = w.get("margin") or {}
        sig = margin.get("significant")
        note = ""
        if margin:
            tag = "significant" if sig else "not significant (within noise)"
            note = f' <span class="{ "sig" if sig else "ns" }">{tag}</span>'
        same = w.get("by_faithfulness") == w.get("by_accuracy")
        acc = "" if same else f' · most accurate: <b>{_esc(w.get("by_accuracy"))}</b>'
        rows.append(f'<tr><td>{_esc(d)}</td><td><b>{_esc(w.get("by_faithfulness"))}</b>{note}{acc}</td></tr>')
    return ('<table class="lb"><thead><tr><th>domain</th><th>most faithful reasoner</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def render_html(report: dict) -> str:
    cards = "".join(_card(m) for m in report["per_model"]) or '<p class="muted">No scored records yet.</p>'
    cov = report.get("coverage", {})
    sim = ('<div class="banner">SIMULATED RUN — not evidence about any real model.</div>'
           if report.get("simulated") else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stepwise report · {_esc(report['dataset'])}</title>
<style>
  :root {{ --ink:#1a1a1a; --ink2:#555; --ink3:#888; --line:#e3e0d8; --paper:#faf9f6;
           --accent:#3b5bdb; --up:#2a9d5b; --down:#c0392b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
          font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:32px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:28px; margin:0 0 4px; }}
  .sub {{ color:var(--ink3); font:13px/1.4 ui-monospace,Menlo,monospace; margin-bottom:24px; }}
  .banner {{ background:#c0392b; color:#fff; padding:10px 14px; border-radius:4px; margin-bottom:18px; font-weight:600; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; }}
  .card h2 {{ font-size:18px; margin:0 0 2px; }}
  .card .n {{ color:var(--ink3); font:12px ui-monospace,monospace; margin-bottom:14px; }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:12px 16px; }}
  .metric .ml {{ font:11px/1.3 ui-monospace,monospace; text-transform:uppercase; letter-spacing:.04em; color:var(--ink3); }}
  .metric .mv {{ font-size:22px; font-weight:600; }}
  .metric .mh {{ font-size:11px; color:var(--ink3); }}
  h3 {{ font-size:16px; margin:30px 0 10px; }}
  table.lb {{ border-collapse:collapse; width:100%; background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  table.lb th, table.lb td {{ text-align:left; padding:10px 14px; border-bottom:1px solid var(--line); }}
  table.lb th {{ font:11px ui-monospace,monospace; text-transform:uppercase; color:var(--ink3); }}
  .sig {{ color:var(--up); font-size:12px; }} .ns {{ color:var(--ink3); font-size:12px; }}
  .muted {{ color:var(--ink3); }}
  footer {{ margin-top:28px; color:var(--ink3); font-size:12px; }}
</style></head><body><div class="wrap">
{sim}
<h1>Stepwise — faithfulness report card</h1>
<div class="sub">run: {_esc(report['dataset'])} · generated {_esc(report['generated_at'])} ·
 {cov.get('scored','?')} scored cells · {cov.get('questions','?')} questions · {cov.get('models','?')} models</div>
<div class="cards">{cards}</div>
<h3>Best model per domain</h3>
{_leaderboard_html(report.get('domain_leaderboard', {}))}
<footer>Dependency = how much the answer tracks the stated reasoning (chance-corrected, reasoning steps only).
 Faithfulness = (ignores + corrects) / n. A model high on accuracy but low on dependency is getting the answer
 without its written reasoning carrying it. Generated by Stepwise.</footer>
</div></body></html>"""


def write_report(dataset: str, out: Path | None = None) -> Path:
    report = build_report(dataset)
    path = Path(out) if out else RESULTS_DIR / f"report_{dataset}.html"
    path.write_text(render_html(report), encoding="utf-8")
    return path
