"""Promotion gate — emits PROMOTE / DO NOT PROMOTE / NEEDS HUMAN REVIEW.

Reads `config/evalops/thresholds.yaml` (via `evalops.metric_config`).
Decision rules:

  PROMOTE
    - all primary metric means are within their threshold (direction-aware)
    - per-item primary failure rate <= primary_failure_rate_threshold
    - no secondary metric is in the failure zone

  DO NOT PROMOTE
    - any primary metric mean crosses its threshold, OR
    - primary failure rate exceeds threshold

  NEEDS HUMAN REVIEW
    - all primaries OK, but at least one secondary metric is in the
      failure zone (mixed signal)

Per the eval-gate spec Invariant 6: this module *recommends* only.
Applying the Langfuse label change is a separate (human or CI) step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from evalops.metric_config import GATE_CONFIG, get as get_metric_config


class Decision(str, Enum):
    PROMOTE = "PROMOTE"
    DO_NOT_PROMOTE = "DO NOT PROMOTE"
    NEEDS_HUMAN_REVIEW = "NEEDS HUMAN REVIEW"


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    reasons: list[str] = field(default_factory=list)


def decide(
    aggregates: dict[str, dict[str, Any]],
    scores: list[dict[str, Any]],
) -> GateResult:
    """Apply gate rules to per-evaluator aggregates + raw score list."""
    primary = list(GATE_CONFIG.get("primary_metrics") or [])
    secondary = list(GATE_CONFIG.get("secondary_metrics") or [])
    failure_rate_threshold = float(
        GATE_CONFIG.get("primary_failure_rate_threshold") or 0.2
    )

    if not aggregates:
        return GateResult(
            Decision.NEEDS_HUMAN_REVIEW,
            ["No scores collected — cannot decide."],
        )

    reasons: list[str] = []
    primary_failures: list[str] = []
    secondary_failures: list[str] = []

    for name in primary:
        agg = aggregates.get(name)
        if agg is None:
            primary_failures.append(f"Primary metric '{name}' has no scores.")
            continue
        cfg = get_metric_config(name)
        if cfg.is_failure(agg["mean"]):
            primary_failures.append(
                f"Primary '{name}' mean {agg['mean']:.3f} crosses threshold "
                f"{cfg.failure_threshold} ({cfg.label()})."
            )

    primary_failure_rate = _failure_rate(scores, primary)
    if primary_failure_rate > failure_rate_threshold:
        primary_failures.append(
            f"Primary per-item failure rate {primary_failure_rate:.1%} "
            f"exceeds {failure_rate_threshold:.0%}."
        )

    for name in secondary:
        agg = aggregates.get(name)
        if agg is None:
            continue
        cfg = get_metric_config(name)
        if cfg.is_failure(agg["mean"]):
            secondary_failures.append(
                f"Secondary '{name}' mean {agg['mean']:.3f} crosses threshold "
                f"{cfg.failure_threshold} ({cfg.label()})."
            )

    if primary_failures:
        return GateResult(Decision.DO_NOT_PROMOTE, primary_failures + _ok_lines(aggregates, primary, secondary))
    if secondary_failures:
        return GateResult(
            Decision.NEEDS_HUMAN_REVIEW,
            secondary_failures + _ok_lines(aggregates, primary, secondary),
        )
    reasons = ["All primary and secondary metrics within thresholds."] + _ok_lines(
        aggregates, primary, secondary
    )
    return GateResult(Decision.PROMOTE, reasons)


def _ok_lines(
    aggregates: dict[str, dict[str, Any]],
    primary: list[str],
    secondary: list[str],
) -> list[str]:
    out: list[str] = []
    for name in primary + secondary:
        agg = aggregates.get(name)
        if agg is None:
            continue
        cfg = get_metric_config(name)
        out.append(f"  - {name}: mean {agg['mean']:.3f} ({cfg.label()}, threshold {cfg.failure_threshold}).")
    return out


def _failure_rate(scores: list[dict[str, Any]], names: list[str]) -> float:
    if not names or not scores:
        return 0.0
    relevant = [s for s in scores if s.get("name") in names]
    if not relevant:
        return 0.0
    fails = 0
    for s in relevant:
        v = s.get("value")
        name = s.get("name")
        if v is None or name is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if get_metric_config(name).is_failure(f):
            fails += 1
    return fails / len(relevant)
