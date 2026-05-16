"""Per-metric semantics: direction + failure threshold.

Direction determines what 'better' means for a numeric score:
  - 'higher_is_better': 1.0 is best, 0.0 is worst (e.g. Correctness)
  - 'lower_is_better':  0.0 is best, 1.0 is worst (e.g. Hallucination)
  - 'unknown':          no failure detection; just report the number

Adding a new evaluator? Add an entry to REGISTRY. Unknown evaluators
fall back to direction='unknown' (no failure flagging).

P2 interim home. Per the eval-gate spec, P3 migrates this config to
`config/evalops/thresholds.yaml` when the promotion gate lands.
"""

from __future__ import annotations

from dataclasses import dataclass

DIRECTION_HIGHER = "higher_is_better"
DIRECTION_LOWER = "lower_is_better"
DIRECTION_UNKNOWN = "unknown"


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


REGISTRY: dict[str, MetricConfig] = {
    "Conciseness": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.5),
    "Hallucination": MetricConfig(direction=DIRECTION_LOWER, failure_threshold=0.5),
    "Correctness": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.9),
    "Helpfulness": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.5),
    "Relevance": MetricConfig(direction=DIRECTION_HIGHER, failure_threshold=0.5),
    "Toxicity": MetricConfig(direction=DIRECTION_LOWER, failure_threshold=0.3),
}

_UNKNOWN = MetricConfig(direction=DIRECTION_UNKNOWN)


def get(name: str) -> MetricConfig:
    return REGISTRY.get(name, _UNKNOWN)
