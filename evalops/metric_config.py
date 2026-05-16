"""Per-metric semantics: direction + failure threshold.

Loaded from `config/evalops/thresholds.yaml` at import time. Code-side
fallback REGISTRY is kept for environments without the YAML (e.g. tests
or repo clones missing config files).

Direction determines what 'better' means for a numeric score:
  - 'higher_is_better': 1.0 is best, 0.0 is worst (e.g. Correctness)
  - 'lower_is_better':  0.0 is best, 1.0 is worst (e.g. Hallucination)
  - 'unknown':          no failure detection; just report the number
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DIRECTION_HIGHER = "higher_is_better"
DIRECTION_LOWER = "lower_is_better"
DIRECTION_UNKNOWN = "unknown"

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "evalops" / "thresholds.yaml"


@dataclass(frozen=True)
class MetricConfig:
    direction: str
    failure_threshold: float | None = None

    def is_failure(self, value: float) -> bool:
        if self.failure_threshold is None or self.direction == DIRECTION_UNKNOWN:
            return False
        if self.direction == DIRECTION_HIGHER:
            return value < self.failure_threshold
        if self.direction == DIRECTION_LOWER:
            return value > self.failure_threshold
        return False

    def label(self) -> str:
        if self.direction == DIRECTION_HIGHER:
            return "higher = better"
        if self.direction == DIRECTION_LOWER:
            return "lower = better"
        return "direction unknown"


_FALLBACK_REGISTRY: dict[str, MetricConfig] = {
    "Conciseness": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.5),
    "Hallucination": MetricConfig(direction=DIRECTION_LOWER, failure_threshold=0.5),
    "Correctness": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.9),
    "Helpfulness": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.5),
    "Relevance": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.5),
    "Toxicity": MetricConfig(direction=DIRECTION_LOWER, failure_threshold=0.3),
}

_UNKNOWN = MetricConfig(direction=DIRECTION_UNKNOWN)


def _load_registry() -> tuple[dict[str, MetricConfig], dict[str, Any]]:
    """Return (registry, gate_config). Falls back to in-code defaults if YAML missing."""
    if not _CONFIG_PATH.exists():
        return dict(_FALLBACK_REGISTRY), {}
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    metrics = data.get("metrics", {}) or {}
    registry = {
        name: MetricConfig(
            direction=spec.get("direction", DIRECTION_UNKNOWN),
            failure_threshold=spec.get("failure_threshold"),
        )
        for name, spec in metrics.items()
    }
    gate = data.get("gate", {}) or {}
    return registry, gate


REGISTRY, GATE_CONFIG = _load_registry()


def get(name: str) -> MetricConfig:
    return REGISTRY.get(name, _UNKNOWN)
