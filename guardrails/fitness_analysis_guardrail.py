"""Structural guardrail for the fitness_analyst task.

A pure-Python function guardrail (no LLM call) that rejects analyst output
which is missing any of the four required sections defined in the task's
expected_output: current state assessment, goal analysis, focus areas, and
recommendations / challenges.

CrewAI's Task guardrail contract: ``(TaskOutput) -> (bool, Any)``. On
success the second element is the validated payload that flows downstream;
on failure it is the rejection reason CrewAI surfaces to the agent for a
retry.

NOTE: we deliberately do NOT annotate the inner `check` closure's return
type. CrewAI's Task.guardrail validator (crewai/task.py) calls
``inspect.signature(check).return_annotation`` and runs ``get_origin`` on
it; with PEP 563 string annotations (``from __future__ import
annotations``) ``get_origin`` returns ``None`` and the validator raises
"If return type is annotated, it must be Tuple[bool, Any]". The outer
builder's annotation still documents the contract for readers.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Tuple

from crewai.tasks.task_output import TaskOutput

# (label, regex). Label is shown in the rejection message; regex is matched
# (case-insensitively, via the lowercased output) to detect the section.
# Patterns allow a few words between the anchor terms so headings like
# "Assessment of Current Fitness State" satisfy the "current state" group.
_REQUIRED_MARKERS = (
    (
        "current state",
        re.compile(r"current\s+(?:\w+\s+){0,3}state|state\s+(?:\w+\s+){0,3}assessment"),
    ),
    ("goal", re.compile(r"\bgoal")),
    ("focus", re.compile(r"\bfocus")),
    ("challenge/recommendation", re.compile(r"\bchallenge|\brecommendation")),
)

_MIN_LENGTH = 300


def build_fitness_analysis_guardrail(
    inputs: Dict[str, Any],
) -> Callable[[TaskOutput], Tuple[bool, Any]]:
    def check(output: TaskOutput):
        text = (output.raw or "").lower()
        if len(text) < _MIN_LENGTH:
            return (
                False,
                f"Fitness profile is too short ({len(text)} chars); expected at "
                f"least {_MIN_LENGTH}. Expand each section with concrete detail.",
            )
        missing = [label for label, pattern in _REQUIRED_MARKERS if not pattern.search(text)]
        if missing:
            return (
                False,
                f"Fitness profile is missing required section(s): {missing}. "
                "Include current state assessment, goal analysis, focus areas, "
                "and recommendations / challenges.",
            )
        return (True, output.raw)

    return check
