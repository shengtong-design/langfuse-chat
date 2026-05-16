"""Eval report generator — Markdown only, local-only.

Renders sections 1-9 of the report contract from the eval-gate spec.
Sections 10-12 (regression analysis, cost/latency, promotion
recommendation) land in P2/P3.

Pure-Python string rendering for P1 — no template engine dependency.
Phase 2 (shape lock) may migrate to Jinja using
`evalops/templates/report.md.j2`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalops.manifest import ExperimentManifest
from evalops.scorer import aggregate, group_by_trace

FAILURE_THRESHOLD = 0.5
PER_ITEM_PREVIEW_LIMIT = 30


def generate_report(
    manifest: ExperimentManifest,
    scores: list[dict[str, Any]],
    reports_dir: Path,
) -> Path:
    """Render a Markdown report for the run; return the written file path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{manifest.experiment_name}.md"
    path.write_text(_render(manifest, scores), encoding="utf-8")
    return path


def _render(manifest: ExperimentManifest, scores: list[dict[str, Any]]) -> str:
    aggregates = aggregate(scores)
    by_trace = group_by_trace(scores)
    failures = _find_failures(scores)

    parts: list[str] = []
    parts.append(f"# {manifest.experiment_name}\n")
    parts.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")

    parts.append(_section_1_summary(aggregates, manifest))
    parts.append(_section_2_config(manifest))
    parts.append(_section_3_dataset(manifest))
    parts.append(_section_4_crew(manifest))
    parts.append(_section_5_prompts(manifest))
    parts.append(_section_6_metric_defs(aggregates))
    parts.append(_section_7_aggregates(aggregates))
    parts.append(_section_8_per_item(by_trace))
    parts.append(_section_9_failures(failures))
    parts.append("---\n_Sections 10 (regression), 11 (cost/latency), 12 (recommendation) pending P2/P3._\n")

    return "\n".join(parts)


def _section_1_summary(aggregates: dict[str, dict[str, Any]], manifest: ExperimentManifest) -> str:
    lines = ["## 1. Executive summary\n"]
    item_count = manifest.dataset.item_count if manifest.dataset else 0
    lines.append(f"- Items evaluated: **{item_count}**")
    if aggregates:
        for name in sorted(aggregates):
            stats = aggregates[name]
            lines.append(
                f"- **{name}**: mean = {stats['mean']:.3f} "
                f"over {stats['count']} samples "
                f"(min {stats['min']:.3f}, max {stats['max']:.3f})"
            )
    else:
        lines.append("- _No scores collected — either no rules fired in window or wait was too short._")
    return "\n".join(lines) + "\n"


def _section_2_config(manifest: ExperimentManifest) -> str:
    return (
        "## 2. Experiment configuration\n\n"
        "```json\n"
        f"{manifest.to_json()}\n"
        "```\n"
    )


def _section_3_dataset(manifest: ExperimentManifest) -> str:
    lines = ["## 3. Dataset\n"]
    if manifest.dataset is None:
        lines.append("- _no dataset captured_")
    else:
        lines.append(f"- Name: `{manifest.dataset.name}`")
        lines.append(f"- Version: `{manifest.dataset.version or '(unspecified)'}`")
        lines.append(f"- Item count: {manifest.dataset.item_count}")
    return "\n".join(lines) + "\n"


def _section_4_crew(manifest: ExperimentManifest) -> str:
    lines = ["## 4. Crew scope\n"]
    if manifest.crew:
        lines.append(f"- Crew: `{manifest.crew.name}` (version `{manifest.crew.version or '(n/a)'}`)")
    if manifest.flow:
        lines.append(f"- Flow: `{manifest.flow.name}` (version `{manifest.flow.version or '(n/a)'}`)")
    if not manifest.crew and not manifest.flow:
        lines.append("- _no crew/flow captured_")
    return "\n".join(lines) + "\n"


def _section_5_prompts(manifest: ExperimentManifest) -> str:
    lines = ["## 5. Prompt versions\n"]
    if not manifest.agent_prompt_versions and not manifest.task_prompt_versions:
        lines.append(f"- Resolved at label: `{manifest.environment or '(unspecified)'}`")
        lines.append("- _Per-agent / per-task version capture lands in P2._")
        return "\n".join(lines) + "\n"
    if manifest.agent_prompt_versions:
        lines.append("**Agents:**")
        for k, v in sorted(manifest.agent_prompt_versions.items()):
            lines.append(f"- `{k}`: label=`{v.label}`, version=`{v.version}`, source=`{v.source}`")
    if manifest.task_prompt_versions:
        lines.append("**Tasks:**")
        for k, v in sorted(manifest.task_prompt_versions.items()):
            lines.append(f"- `{k}`: label=`{v.label}`, version=`{v.version}`, source=`{v.source}`")
    return "\n".join(lines) + "\n"


def _section_6_metric_defs(aggregates: dict[str, dict[str, Any]]) -> str:
    lines = ["## 6. Evaluation metrics\n"]
    if not aggregates:
        lines.append("- _No metrics fired._")
        return "\n".join(lines) + "\n"
    lines.append("Metrics are Langfuse LLM-as-a-Judge evaluators. Definitions live in Langfuse Cloud (see Settings → Evaluators).\n")
    for name in sorted(aggregates):
        lines.append(f"- **{name}** — numeric 0-1 (managed evaluator)")
    return "\n".join(lines) + "\n"


def _section_7_aggregates(aggregates: dict[str, dict[str, Any]]) -> str:
    lines = ["## 7. Aggregate scores\n"]
    if not aggregates:
        lines.append("_No scores collected._")
        return "\n".join(lines) + "\n"
    lines.append("| Metric | Count | Mean | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in sorted(aggregates):
        s = aggregates[name]
        lines.append(f"| {name} | {s['count']} | {s['mean']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
    return "\n".join(lines) + "\n"


def _section_8_per_item(by_trace: dict[str, dict[str, float]]) -> str:
    lines = ["## 8. Per-item results\n"]
    if not by_trace:
        lines.append("_No per-trace scores collected._")
        return "\n".join(lines) + "\n"
    metric_names = sorted({m for t in by_trace.values() for m in t})
    header = "| trace_id | " + " | ".join(metric_names) + " |"
    sep = "|---|" + "|".join(["---:"] * len(metric_names)) + "|"
    lines.append(header)
    lines.append(sep)
    trace_ids = list(by_trace.keys())
    for tid in trace_ids[:PER_ITEM_PREVIEW_LIMIT]:
        row = by_trace[tid]
        cells = [f"{row[m]:.3f}" if m in row else "—" for m in metric_names]
        lines.append(f"| `{tid[:12]}…` | " + " | ".join(cells) + " |")
    if len(trace_ids) > PER_ITEM_PREVIEW_LIMIT:
        lines.append(f"\n_{len(trace_ids) - PER_ITEM_PREVIEW_LIMIT} more rows omitted from preview._")
    return "\n".join(lines) + "\n"


def _section_9_failures(failures: list[dict[str, Any]]) -> str:
    lines = ["## 9. Failure examples\n"]
    lines.append(f"_Threshold: score < {FAILURE_THRESHOLD}_\n")
    if not failures:
        lines.append("None.")
        return "\n".join(lines) + "\n"
    lines.append("| trace_id | metric | value | comment |")
    lines.append("|---|---|---:|---|")
    for f in failures[:PER_ITEM_PREVIEW_LIMIT]:
        tid = (f.get("traceId") or "")[:12]
        name = f.get("name", "")
        val = f.get("value")
        comment = (f.get("comment") or "").replace("|", "\\|").replace("\n", " ")[:120]
        val_str = f"{float(val):.3f}" if val is not None else "—"
        lines.append(f"| `{tid}…` | {name} | {val_str} | {comment} |")
    if len(failures) > PER_ITEM_PREVIEW_LIMIT:
        lines.append(f"\n_{len(failures) - PER_ITEM_PREVIEW_LIMIT} more failures omitted from preview._")
    return "\n".join(lines) + "\n"


def _find_failures(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fails = []
    for s in scores:
        v = s.get("value")
        if v is None:
            continue
        try:
            if float(v) < FAILURE_THRESHOLD:
                fails.append(s)
        except (TypeError, ValueError):
            continue
    return fails
