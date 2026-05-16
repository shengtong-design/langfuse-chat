"""Flow execution adapter — wraps CrewAI flows for dataset-item execution.

Per the eval-gate spec Invariant 3: eval runs hit the same `flow.kickoff()`
entry point as production. This module is a thin adapter; it never
assembles its own LLM glue.
"""

from __future__ import annotations

from typing import Any, Callable

from core.observability import ConnectorManager
from crews.common import extract_question
from flows.research_flow import ResearchFlow

CREW_FLOWS: dict[str, type] = {
    "researcher": ResearchFlow,
}


def get_flow_class(crew_name: str) -> type:
    if crew_name not in CREW_FLOWS:
        known = sorted(CREW_FLOWS)
        raise ValueError(f"Unknown crew '{crew_name}'. Known crews: {known}")
    return CREW_FLOWS[crew_name]


def make_task(crew_name: str, connectors: ConnectorManager) -> Callable[[Any], str]:
    """Build the per-item task callable that `langfuse.run_experiment` will invoke."""
    flow_cls = get_flow_class(crew_name)

    def task(item: Any) -> str:
        q = extract_question(item.input)
        flow = flow_cls(connectors_factory=lambda: connectors)
        result = flow.kickoff(inputs={"question": q})
        return result.get("result", "") if isinstance(result, dict) else str(result)

    return task
