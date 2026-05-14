"""Structural guardrail for the fitness_analyst task.

A pure-Python function guardrail (no LLM call) that rejects analyst output
which is missing any of the four required sections defined in the task's
expected_output: current state assessment, goal analysis, focus areas, and
recommendations / challenges.

CrewAI's Task guardrail contract: ``(TaskOutput) -> (bool, Any)``. On
success the second element is the validated payload that flows downstream;
on failure it is the rejection reason CrewAI surfaces to the agent for a
retry.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from crewai.tasks.task_output import TaskOutput

_REQUIRED_MARKERS = (
    ("current state", "state assessment"),
    ("goal",),
    ("focus",),
    ("challenge", "recommendation"),
)

_MIN_LENGTH = 300


def build_fitness_analysis_guardrail(
    inputs: Dict[str, Any],
) -> Callable[[TaskOutput], Tuple[bool, Any]]:
    def check(output: TaskOutput) -> Tuple[bool, Any]:
        text = (output.raw or "").lower()
        if len(text) < _MIN_LENGTH:
            return (
                False,
                f"Fitness profile is too short ({len(text)} chars); expected at "
                f"least {_MIN_LENGTH}. Expand each section with concrete detail.",
            )
        missing = [group[0] for group in _REQUIRED_MARKERS if not any(m in text for m in group)]
        if missing:
            return (
                False,
                f"Fitness profile is missing required section(s): {missing}. "
                "Include current state assessment, goal analysis, focus areas, "
                "and recommendations / challenges.",
            )
        return (True, output.raw)

    return check
