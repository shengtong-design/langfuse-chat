"""Guardrails package — CrewAI task guardrail instances + registry.

Tasks reference guardrails by string key in their YAML
(e.g. ``guardrail: fitness_analysis_guardrail``). The key is resolved against
GUARDRAIL_BUILDERS in crews/base.py at task-build time: each builder
receives the run's inputs dict and returns a freshly-constructed
guardrail, so per-run state (uploaded files, user goals, ...) can be
injected into the validator's reference context without leaking through
Langfuse-editable prompt text.

Add a new guardrail:
    1. Implement a builder in guardrails/<name>_guardrail.py.
    2. Register it here under a unique snake_case key.
    3. Reference the key from the relevant task's YAML under ``guardrail:``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .fitness_analysis_guardrail import build_fitness_analysis_guardrail

GUARDRAIL_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "fitness_analysis_guardrail": build_fitness_analysis_guardrail,
}

__all__ = ["GUARDRAIL_BUILDERS", "build_fitness_analysis_guardrail"]
