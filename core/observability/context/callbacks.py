"""CrewAI callback handlers.

Wire to a Crew like this:

    from core.observability.context.callbacks import get_crew_kwargs

    crew = Crew(agents=[...], tasks=[...], **get_crew_kwargs(obs))

get_crew_kwargs() is a no-op when obs is a plain ConnectorManager, so the crew
files don't need to branch on whether the addon is active.
"""
import logging
from typing import Any

log = logging.getLogger(__name__)


class CrewCallbacks:
    """step_callback and task_callback implementations for CrewAI Crew()."""

    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def on_agent_step(self, step_output: Any) -> None:
        """Fires after each agent reasoning step (thought / tool use / observation)."""
        try:
            output_str = str(step_output)[:2000]
            with self._obs.span(
                "agent.step", "agent",
                input_data={"step": output_str},
            ) as h:
                h.update(output={"step": output_str})
        except Exception:
            log.debug("agent.step span failed", exc_info=True)

    def on_task_complete(self, task_output: Any) -> None:
        """Fires when each CrewAI Task finishes."""
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


def get_crew_kwargs(obs: Any) -> dict:
    """Return step/task callback kwargs for Crew() if the addon is active, else {}."""
    if hasattr(obs, "crew_callbacks"):
        cb = obs.crew_callbacks
        return {
            "step_callback": cb.on_agent_step,
            "task_callback": cb.on_task_complete,
        }
    return {}
