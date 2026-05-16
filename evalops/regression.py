"""Regression analysis — compare current run's aggregates against a baseline.

The baseline is the most recent prior `production`-label run for the same
crew + dataset. Per-metric deltas are direction-aware: a regression for a
higher-is-better metric means the mean dropped; for lower-is-better it
means the mean rose.

Source of aggregates: the manifest's `aggregates` field (schema_version
>= 1.1). Older manifests are skipped — they're not comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalops.manifest import CrewRef, DatasetRef, ExperimentManifest
from evalops.metric_config import DIRECTION_HIGHER, DIRECTION_LOWER, get as get_metric_config

DEFAULT_TOLERANCE = 0.05


@dataclass(frozen=True)
class MetricDelta:
    name: str
    direction: str
    current_mean: float
    baseline_mean: float
    delta: float
    is_regression: bool
    is_improvement: bool


@dataclass(frozen=True)
class RegressionReport:
    baseline_experiment: str
    baseline_started_at: str
    tolerance: float
    deltas: list[MetricDelta]

    @property
    def has_regression(self) -> bool:
        return any(d.is_regression for d in self.deltas)


def find_baseline(
    manifests_dir: Path,
    current: ExperimentManifest,
    *,
    target_environment: str | None = None,
) -> ExperimentManifest | None:
    """Return the most recent prior manifest for the same crew + dataset + environment.

    By default `target_environment` follows `current.environment`, so staging
    runs compare against staging baselines and production against production.
    Pass an explicit value to override (e.g. always compare against production).
    """
    if current.crew is None or current.dataset is None:
        return None

    if target_environment is None:
        target_environment = current.environment

    candidates: list[tuple[str, ExperimentManifest]] = []
    for path in manifests_dir.glob("*.json"):
        if path.name == f"{current.experiment_name}.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not data.get("aggregates"):
            continue
        crew = (data.get("crew") or {}).get("name")
        dataset = (data.get("dataset") or {}).get("name")
        env = data.get("environment")
        started = data.get("started_at")
        if (
            crew != current.crew.name
            or dataset != current.dataset.name
            or env != target_environment
            or not started
            or started >= current.started_at
        ):
            continue
        candidates.append((started, _hydrate(data)))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def compare(
    current: ExperimentManifest,
    baseline: ExperimentManifest,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> RegressionReport:
    """Compute per-metric deltas + regression flags."""
    deltas: list[MetricDelta] = []
    for name in sorted(current.aggregates.keys() | baseline.aggregates.keys()):
        cur = (current.aggregates.get(name) or {}).get("mean")
        base = (baseline.aggregates.get(name) or {}).get("mean")
        if cur is None or base is None:
            continue
        cfg = get_metric_config(name)
        delta = float(cur) - float(base)
        is_regression = False
        is_improvement = False
        if abs(delta) >= tolerance:
            if cfg.direction == DIRECTION_HIGHER:
                is_regression = delta < 0
                is_improvement = delta > 0
            elif cfg.direction == DIRECTION_LOWER:
                is_regression = delta > 0
                is_improvement = delta < 0
        deltas.append(
            MetricDelta(
                name=name,
                direction=cfg.direction,
                current_mean=float(cur),
                baseline_mean=float(base),
                delta=delta,
                is_regression=is_regression,
                is_improvement=is_improvement,
            )
        )

    return RegressionReport(
        baseline_experiment=baseline.experiment_name,
        baseline_started_at=baseline.started_at,
        tolerance=tolerance,
        deltas=deltas,
    )


def _hydrate(data: dict[str, Any]) -> ExperimentManifest:
    """Reconstruct an ExperimentManifest from a JSON dict for read-only use.

    We only populate the fields the regression module needs; full
    round-trip fidelity isn't required here.
    """
    from evalops.manifest import FlowRef

    crew = CrewRef(**data["crew"]) if data.get("crew") else None
    dataset = DatasetRef(**data["dataset"]) if data.get("dataset") else None
    flow = FlowRef(**data["flow"]) if data.get("flow") else None
    return ExperimentManifest(
        experiment_name=data["experiment_name"],
        started_at=data["started_at"],
        completed_at=data.get("completed_at"),
        dataset=dataset,
        crew=crew,
        flow=flow,
        environment=data.get("environment"),
        metrics_requested=list(data.get("metrics_requested") or []),
        aggregates=dict(data.get("aggregates") or {}),
    )


def find_baseline_in_langfuse(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    current: ExperimentManifest,
) -> ExperimentManifest | None:
    """Fallback baseline lookup via Langfuse when local manifests are absent.

    Streamlit Cloud wipes `evalops/manifests/` on every redeploy, so
    `find_baseline()` against the local FS often misses. Langfuse, by
    contrast, persists every dataset run with the metadata EvalOps wrote
    on `run_experiment`. We re-derive aggregates from scores in the
    appropriate time window — best-effort and never raises.
    """
    if current.dataset is None or current.crew is None:
        return None

    from evalops.scorer import aggregate, fetch_dataset_runs, fetch_scores

    try:
        runs = fetch_dataset_runs(
            base_url=base_url,
            public_key=public_key,
            secret_key=secret_key,
            dataset_name=current.dataset.name,
        )
    except Exception:
        return None

    candidates: list[dict[str, Any]] = []
    for r in runs:
        if r.get("name") == current.experiment_name:
            continue
        md = r.get("metadata") or {}
        if md.get("crew") and md["crew"] != current.crew.name:
            continue
        if md.get("prompt_label") and md["prompt_label"] != current.environment:
            continue
        created = r.get("createdAt")
        if not created or created >= current.started_at:
            continue
        candidates.append(r)

    if not candidates:
        return None

    candidates.sort(key=lambda r: r.get("createdAt") or "", reverse=True)
    baseline_run = candidates[0]
    baseline_started = baseline_run["createdAt"]

    # Bound the score window: from baseline.createdAt until the next-later
    # run for the same dataset (or until the current run's start).
    later = [
        r.get("createdAt")
        for r in runs
        if r.get("createdAt") and r["createdAt"] > baseline_started
    ]
    to_ts = min(later) if later else current.started_at

    try:
        scores = fetch_scores(
            base_url=base_url,
            public_key=public_key,
            secret_key=secret_key,
            from_timestamp=baseline_started,
            to_timestamp=to_ts,
            evaluator_names=current.metrics_requested or None,
        )
    except Exception:
        return None

    return ExperimentManifest(
        experiment_name=baseline_run["name"],
        started_at=baseline_started,
        crew=CrewRef(name=current.crew.name),
        dataset=DatasetRef(name=current.dataset.name),
        environment=current.environment,
        metrics_requested=list(current.metrics_requested or []),
        aggregates=aggregate(scores),
    )
