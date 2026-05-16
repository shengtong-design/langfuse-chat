"""LLM-as-a-Judge evaluator selection and score collection.

Phase 0 stub. Phase 1 implementation will:
- Enumerate configured LLM-as-a-Judge evaluators in Langfuse Cloud.
- Trigger / wait for / collect scores for an experiment run.
- Return per-item and aggregate scores keyed by evaluator name.

Per the eval-gate spec Invariant 8: this module never authors scoring
logic. Custom metrics are created in Langfuse first, then referenced here
by name.
"""

from __future__ import annotations


def list_configured_evaluators() -> list[str]:
    """Return the names of LLM-as-a-Judge evaluators available in Langfuse."""
    raise NotImplementedError("Phase 1: implement Langfuse LLM-as-a-Judge evaluator lookup.")
