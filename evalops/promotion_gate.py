"""Promotion gate — emits PROMOTE / DO NOT PROMOTE / NEEDS HUMAN REVIEW.

Phase 0 stub. Phase 3 implementation reads thresholds from
`config/evalops/thresholds.yaml` and applies them to aggregate scores
plus regression deltas against the previous production run.

Per the eval-gate spec Invariant 6: this module *recommends* only;
applying the label change in Langfuse is a separate (human or CI) step.
"""

from __future__ import annotations

from enum import Enum


class Decision(str, Enum):
    PROMOTE = "PROMOTE"
    DO_NOT_PROMOTE = "DO NOT PROMOTE"
    NEEDS_HUMAN_REVIEW = "NEEDS HUMAN REVIEW"


def decide(*args, **kwargs) -> Decision:
    raise NotImplementedError("Phase 3: implement promotion gate logic.")
