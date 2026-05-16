"""Eval report generator — Markdown only, local-only.

Renders sections 1-9 of the report contract from the eval-gate spec
with real content; sections 10-12 are declared as placeholders so the
report shape is locked across runs (P2 contract).

Pure-Python string rendering — no template engine dependency. Phase 3+
may migrate to Jinja using `evalops/templates/report.md.j2`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalops.flow_introspect import FlowGraph, render_mermaid, render_text_tree
from evalops.manifest import ExperimentManifest
from evalops.metric_config import (
    DIRECTION_HIGHER,
    DIRECTION_LOWER,
    get as get_metric_config,
)
from evalops.promotion_gate import decide as gate_decide
from evalops.regression import RegressionReport
from evalops.scorer import aggregate, group_by_trace

PER_ITEM_PREVIEW_LIMIT = 30


def generate_report(
    manifest: ExperimentManifest,
    scores: list[dict[str, Any]],
    reports_dir: Path,
    trace_labels: dict[str, str] | None = None,
    regression: RegressionReport | None = None,
    flow_graph: FlowGraph | None = None,
    trace_expected: dict[str, str] | None = None,
    trace_actual: dict[str, str] | None = None,
) -> Path:
    """Render a Markdown report for the run; return the written file path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{manifest.experiment_name}.md"
    path.write_text(
        _render(
            manifest,
            scores,
            trace_labels or {},
            regression,
            flow_graph,
            trace_expected or {},
            trace_actual or {},
        ),
        encoding="utf-8",
    )
    return path


def _render(
    manifest: ExperimentManifest,
    scores: list[dict[str, Any]],
    trace_labels: dict[str, str],
    regression: RegressionReport | None,
    flow_graph: FlowGraph | None,
    trace_expected: dict[str, str],
    trace_actual: dict[str, str],
) -> str:
    aggregates = aggregate(scores)
    by_trace = group_by_trace(scores)
    failures = _find_failures(scores)

    parts: list[str] = []
    parts.append(f"# {manifest.experiment_name}\n")
    parts.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_  ")
    parts.append(f"_Schema: {manifest.schema_version}_\n")

    parts.append(_section_1_summary(aggregates, manifest))
    parts.append(_section_2_config(manifest))
    parts.append(_section_3_dataset(manifest))
    parts.append(_section_4_flow_architecture(manifest, flow_graph))
    parts.append(_section_5_prompts(manifest))
    parts.append(_section_6_metric_defs(aggregates))
    parts.append(_section_7_aggregates(aggregates))
    parts.append(_section_8_per_item(by_trace, trace_labels, trace_expected))
    parts.append(_section_9_failures(failures, trace_labels, trace_expected, trace_actual))
    parts.append(_section_10_regression(regression))
    parts.append(_section_11_cost_latency())
    parts.append(_section_12_recommendation(aggregates, scores))

    return "\n".join(parts)


def _format_item(tid: str, trace_labels: dict[str, str]) -> str:
    label = (trace_labels.get(tid) or "").strip()
    if label:
        safe = label.replace("|", "\\|")
        return f"{safe} `{tid[:8]}`"
    return f"`{tid[:12]}…`"


def _direction_arrow(direction: str) -> str:
    if direction == DIRECTION_HIGHER:
        return "↑"
    if direction == DIRECTION_LOWER:
        return "↓"
    return "?"


def _section_1_summary(aggregates: dict[str, dict[str, Any]], manifest: ExperimentManifest) -> str:
    lines = ["## 1. Executive summary\n"]
    item_count = manifest.dataset.item_count if manifest.dataset else 0
    lines.append(f"- Items evaluated: **{item_count}**")
    if aggregates:
        for name in sorted(aggregates):
            stats = aggregates[name]
            cfg = get_metric_config(name)
            arrow = _direction_arrow(cfg.direction)
            lines.append(
                f"- **{name}** {arrow}: mean = {stats['mean']:.3f} "
                f"over {stats['count']} samples "
                f"(min {stats['min']:.3f}, max {stats['max']:.3f}) — {cfg.label()}"
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


def _section_4_flow_architecture(
    manifest: ExperimentManifest,
    flow_graph: FlowGraph | None,
) -> str:
    lines = ["## 4. Flow architecture\n"]
    if manifest.crew:
        lines.append(f"- Crew: `{manifest.crew.name}` (crew_version `{manifest.crew.version or '(n/a)'}`)")
    if manifest.flow:
        lines.append(f"- Flow: `{manifest.flow.name}` (flow_version `{manifest.flow.version or '(n/a)'}`)")
    if not manifest.crew and not manifest.flow:
        lines.append("- _no crew/flow captured_")

    if flow_graph is None:
        lines.append("\n_Flow introspection not available for this run._")
        return "\n".join(lines) + "\n"

    lines.append("\n### Mermaid diagram\n")
    lines.append("```mermaid")
    lines.append(render_mermaid(flow_graph))
    lines.append("```\n")

    lines.append("### Structure\n")
    lines.append("```")
    lines.append(render_text_tree(flow_graph))
    lines.append("```")

    if flow_graph.notes:
        lines.append("\n_Introspection notes:_")
        for n in flow_graph.notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


def _section_5_prompts(manifest: ExperimentManifest) -> str:
    lines = ["## 5. Prompt versions\n"]
    if not manifest.agent_prompt_versions and not manifest.task_prompt_versions:
        lines.append(f"- Resolved at label: `{manifest.environment or '(unspecified)'}`")
        lines.append("- _Per-agent / per-task version capture: pending PromptLoader hook (deferred)._")
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
    lines.append("Metrics are Langfuse LLM-as-a-Judge evaluators. Definitions live in Langfuse Cloud (see Settings → Evaluators). Direction + failure threshold are configured in `evalops/metric_config.py`.\n")
    lines.append("| Metric | Direction | Failure threshold | Output |")
    lines.append("|---|---|---:|---|")
    for name in sorted(aggregates):
        cfg = get_metric_config(name)
        thr = f"{cfg.failure_threshold:.2f}" if cfg.failure_threshold is not None else "—"
        lines.append(f"| {name} | {cfg.label()} | {thr} | numeric 0–1 |")
    return "\n".join(lines) + "\n"


def _section_7_aggregates(aggregates: dict[str, dict[str, Any]]) -> str:
    lines = ["## 7. Aggregate scores\n"]
    if not aggregates:
        lines.append("_No scores collected._")
        return "\n".join(lines) + "\n"
    lines.append("| Metric | Dir | Count | Mean | Min | Max |")
    lines.append("|---|:-:|---:|---:|---:|---:|")
    for name in sorted(aggregates):
        s = aggregates[name]
        cfg = get_metric_config(name)
        arrow = _direction_arrow(cfg.direction)
        lines.append(f"| {name} | {arrow} | {s['count']} | {s['mean']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
    return "\n".join(lines) + "\n"


def _truncate(text: str, max_len: int) -> str:
    text = (text or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text or "—"


def _section_8_per_item(
    by_trace: dict[str, dict[str, float]],
    trace_labels: dict[str, str],
    trace_expected: dict[str, str],
) -> str:
    lines = ["## 8. Per-item results\n"]
    if not by_trace:
        lines.append("_No per-trace scores collected._")
        return "\n".join(lines) + "\n"
    metric_names = sorted({m for t in by_trace.values() for m in t})
    header_metrics = " | ".join(f"{m} {_direction_arrow(get_metric_config(m).direction)}" for m in metric_names)
    header = "| Item | Expected | " + header_metrics + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(metric_names)) + "|"
    lines.append(header)
    lines.append(sep)

    def _trace_sort_key(tid: str) -> tuple[int, str]:
        label = trace_labels.get(tid, "")
        return (0, label) if label else (1, tid)

    trace_ids = sorted(by_trace.keys(), key=_trace_sort_key)
    for tid in trace_ids[:PER_ITEM_PREVIEW_LIMIT]:
        row = by_trace[tid]
        cells = [f"{row[m]:.3f}" if m in row else "—" for m in metric_names]
        expected = _truncate(trace_expected.get(tid, ""), 60)
        lines.append(f"| {_format_item(tid, trace_labels)} | {expected} | " + " | ".join(cells) + " |")
    if len(trace_ids) > PER_ITEM_PREVIEW_LIMIT:
        lines.append(f"\n_{len(trace_ids) - PER_ITEM_PREVIEW_LIMIT} more rows omitted from preview._")
    return "\n".join(lines) + "\n"


def _section_9_failures(
    failures: list[dict[str, Any]],
    trace_labels: dict[str, str],
    trace_expected: dict[str, str],
    trace_actual: dict[str, str],
) -> str:
    lines = ["## 9. Failure examples\n"]
    lines.append("_Failure: score crosses the metric's direction-specific threshold (see Section 6)._\n")
    if not failures:
        lines.append("None.")
        return "\n".join(lines) + "\n"
    failures_sorted = sorted(failures, key=lambda f: (f.get("name") or "", f.get("traceId") or ""))
    lines.append("| Item | metric | value | Expected | Actual | Judge comment |")
    lines.append("|---|---|---:|---|---|---|")
    for f in failures_sorted[:PER_ITEM_PREVIEW_LIMIT]:
        tid = f.get("traceId") or ""
        name = f.get("name", "")
        val = f.get("value")
        val_str = f"{float(val):.3f}" if val is not None else "—"
        expected = _truncate(trace_expected.get(tid, ""), 80)
        actual = _truncate(trace_actual.get(tid, ""), 80)
        comment = _truncate(f.get("comment") or "", 100)
        lines.append(
            f"| {_format_item(tid, trace_labels)} | {name} | {val_str} | "
            f"{expected} | {actual} | {comment} |"
        )
    if len(failures_sorted) > PER_ITEM_PREVIEW_LIMIT:
        lines.append(f"\n_{len(failures_sorted) - PER_ITEM_PREVIEW_LIMIT} more failures omitted from preview._")
    return "\n".join(lines) + "\n"


def _section_10_regression(regression: RegressionReport | None) -> str:
    lines = ["## 10. Regression analysis\n"]
    if regression is None:
        lines.append("_No prior `production` run found for this crew + dataset — no baseline to compare against._")
        return "\n".join(lines) + "\n"
    lines.append(
        f"Baseline: `{regression.baseline_experiment}` ({regression.baseline_started_at}).  "
        f"Tolerance: ±{regression.tolerance:.2f}.\n"
    )
    lines.append("| Metric | Baseline | Current | Δ | Verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for d in regression.deltas:
        sign = "+" if d.delta > 0 else ("" if d.delta == 0 else "")
        delta_str = f"{sign}{d.delta:+.3f}".replace("++", "+")
        if d.is_regression:
            verdict = "⚠️ regression"
        elif d.is_improvement:
            verdict = "✅ improvement"
        else:
            verdict = "≈ within tolerance"
        lines.append(
            f"| {d.name} {_direction_arrow(d.direction)} "
            f"| {d.baseline_mean:.3f} "
            f"| {d.current_mean:.3f} "
            f"| {delta_str} "
            f"| {verdict} |"
        )
    if regression.has_regression:
        lines.append("\n_At least one metric crossed the regression tolerance._")
    return "\n".join(lines) + "\n"


def _section_11_cost_latency() -> str:
    return (
        "## 11. Cost / latency\n\n"
        "_Pending P6 — populated from Datadog after [[architecture-roadmap]] Q4 (Datadog scope) resolves._\n"
    )


def _section_12_recommendation(
    aggregates: dict[str, dict[str, Any]],
    scores: list[dict[str, Any]],
) -> str:
    from evalops.metric_config import GATE_CONFIG

    result = gate_decide(aggregates, scores)
    icon = {
        "PROMOTE": "✅",
        "DO NOT PROMOTE": "❌",
        "NEEDS HUMAN REVIEW": "⚠️",
    }.get(result.decision.value, "•")

    primary = list(GATE_CONFIG.get("primary_metrics") or [])
    secondary = list(GATE_CONFIG.get("secondary_metrics") or [])

    def _metric_line(name: str) -> str:
        cfg = get_metric_config(name)
        agg = aggregates.get(name)
        if agg is None:
            return f"- ⚠ **{name}**: no scores collected"
        mean = agg["mean"]
        failed = cfg.is_failure(mean)
        mark = "✗" if failed else "✓"
        rel = "<" if cfg.direction == DIRECTION_HIGHER else ">"
        thr = cfg.failure_threshold
        cmp = f"crossed threshold {thr:.2f}" if failed else f"within threshold {thr:.2f}"
        return f"- {mark} **{name}**: {mean:.3f} ({cfg.label()}) — {cmp}"

    lines = ["## 12. Promotion recommendation\n"]
    lines.append(f"### Decision\n\n**{icon} {result.decision.value}**\n")

    lines.append("### Primary metrics (gate)\n")
    if primary:
        for name in primary:
            lines.append(_metric_line(name))
    else:
        lines.append("- _no primary metrics configured_")
    lines.append("")

    lines.append("### Secondary metrics\n")
    if secondary:
        for name in secondary:
            if aggregates.get(name) is None:
                continue
            lines.append(_metric_line(name))
    else:
        lines.append("- _no secondary metrics configured_")
    lines.append("")

    lines.append("### Decision rule applied\n")
    if result.decision.value == "PROMOTE":
        rule = "All primary metrics and per-item failure rate within thresholds. No secondary metric flagged."
    elif result.decision.value == "DO NOT PROMOTE":
        rule = ("At least one primary metric mean or the per-item failure rate "
                "crossed its threshold. Promotion is automatically blocked.")
    else:
        rule = ("All primary metrics pass, but a secondary metric flagged a concern. "
                "Per gate rules, this triggers human review rather than auto-rejection.")
    lines.append(f"_{rule}_\n")

    lines.append("### Suggested next action\n")
    if result.decision.value == "PROMOTE":
        action = ("Promote this prompt/crew configuration to the production label in "
                  "Langfuse. Capture the manifest hash in your release notes.")
    elif result.decision.value == "DO NOT PROMOTE":
        action = ("Do not promote. Review the failing primary metric(s) and the failure "
                  "examples in Section 9. Iterate on the prompt or crew configuration "
                  "and re-run before considering promotion again.")
    else:
        flagged = [
            name for name in secondary
            if aggregates.get(name) is not None and get_metric_config(name).is_failure(aggregates[name]["mean"])
        ]
        if flagged:
            action = (
                f"Have a human reviewer look at the flagged secondary metric(s): "
                f"**{', '.join(flagged)}**. Inspect the per-item table (Section 8) "
                f"and failure examples (Section 9) for the qualitative pattern, then "
                f"decide whether to promote or iterate."
            )
        else:
            action = ("Have a human reviewer inspect the report before deciding.")
    lines.append(action)
    lines.append("")

    lines.append("_Rules in `config/evalops/thresholds.yaml`. "
                 "Section 10 covers regression vs the prior run for the same crew + dataset + environment._")
    return "\n".join(lines) + "\n"


def _find_failures(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fails = []
    for s in scores:
        v = s.get("value")
        name = s.get("name")
        if v is None or name is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        cfg = get_metric_config(name)
        if cfg.is_failure(f):
            fails.append(s)
    return fails
