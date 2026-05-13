"""CrewAI callback handlers.

Wire to a Crew like this:

    from core.observability.context.callbacks import get_crew_kwargs

    crew = Crew(agents=[...], tasks=[...], **get_crew_kwargs(obs))

get_crew_kwargs() is a no-op when obs is a plain ConnectorManager, so the crew
files don't need to branch on whether the addon is active.
"""
import logging
from typing import Any, Dict, Optional, Tuple, Union

log = logging.getLogger(__name__)


_CREWAI_PARSE_FAILURES = {"failed to parse llm response", "could not parse llm output"}


def _clean_thought(raw: str) -> tuple[str, bool]:
    """Return (thought, is_parse_failure). Detects CrewAI internal parse-error messages."""
    cleaned = raw.strip()
    is_failure = cleaned.lower() in _CREWAI_PARSE_FAILURES
    return cleaned, is_failure


def _parse_step(step: Any) -> Tuple[Dict, Optional[Dict]]:
    """Extract structured input/output from a CrewAI AgentAction or AgentFinish."""
    type_name = type(step).__name__

    if type_name == "AgentFinish":
        raw_thought = str(getattr(step, "thought", "") or "")[:500]
        thought, is_parse_failure = _clean_thought(raw_thought)
        raw_output = getattr(step, "output", None) or getattr(step, "return_values", {})
        if isinstance(raw_output, dict):
            raw_output = raw_output.get("output", str(raw_output))
        input_data = {} if is_parse_failure else {"thought": thought}
        output: Dict[str, Any] = {"result": str(raw_output)[:2000]}
        if is_parse_failure:
            output["parse_warning"] = "crewai_failed_to_parse_llm_response"
        return input_data, output

    if type_name == "AgentAction":
        raw_thought = str(getattr(step, "thought", "") or getattr(step, "log", "") or "")[:500]
        thought, _ = _clean_thought(raw_thought)
        return (
            {
                "thought": thought,
                "tool": str(getattr(step, "tool", "")),
                "tool_input": str(getattr(step, "tool_input", ""))[:500],
            },
            None,
        )

    return ({"step_type": type_name}, {"raw": str(step)[:500]})


class _StepCallback:
    """Callable wrapper so Pydantic sees a serialisable type, not a bound method."""
    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def __call__(self, step_output: Any) -> None:
        try:
            input_data, output = _parse_step(step_output)
            with self._obs.span("agent.step", "agent", input_data=input_data) as h:
                if output is not None:
                    h.update(output=output)
        except Exception:
            log.debug("agent.step span failed", exc_info=True)


class _TaskCallback:
    """Callable wrapper so Pydantic sees a serialisable type, not a bound method."""
    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def __call__(self, task_output: Any) -> None:
        try:
            result_str = str(getattr(task_output, "raw", task_output))[:2000]
            description = str(getattr(task_output, "description", ""))[:500]
            agent_role = str(getattr(task_output, "agent", ""))[:100]
            with self._obs.span(
                "task.complete", "span",
                input_data={"description": description, "agent": agent_role},
            ) as h:
                h.update(output={"result": result_str})
        except Exception:
            log.debug("task.complete span failed", exc_info=True)


class CrewCallbacks:
    def __init__(self, obs: Any) -> None:
        self.on_agent_step = _StepCallback(obs)
        self.on_task_complete = _TaskCallback(obs)


def get_crew_kwargs(obs: Any) -> dict:
    """Return step/task callback kwargs for Crew() if the addon is active, else {}."""
    if hasattr(obs, "crew_callbacks"):
        cb = obs.crew_callbacks
        return {
            "step_callback": cb.on_agent_step,
            "task_callback": cb.on_task_complete,
        }
    return {}
