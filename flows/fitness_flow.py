from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from crewai.flow.flow import Flow, start
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class FitnessState(BaseModel):
    goals: str = ""
    fitness_level: str = "beginner"
    equipment: str = ""
    time_per_week: int = 5
    limitations: str = "None specified"
    result: str = ""
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


class FitnessFlow(Flow[FitnessState]):
    """Orchestrates the FitnessCrew to produce a personalized fitness plan.

    Usage:
        flow = FitnessFlow(connectors_factory=_get_connectors)
        result = flow.kickoff(inputs={"goals": "...", "fitness_level": "beginner", ...})
    """

    def __init__(
        self,
        connectors_factory: Callable,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._connectors_factory = connectors_factory

    @start()
    def run_fitness_plan(self) -> Dict[str, Any]:
        try:
            from core.observability.context import EnrichedConnectorManager, make_run_context
            from crews.fitness_crew import FitnessCrew

            inputs = {
                "goals": self.state.goals,
                "fitness_level": self.state.fitness_level,
                "equipment": self.state.equipment,
                "time_per_week": self.state.time_per_week,
                "limitations": self.state.limitations,
            }
            obs = EnrichedConnectorManager(
                self._connectors_factory(),
                make_run_context("fitness_training"),
            )
            data = FitnessCrew().run(
                inputs,
                obs,
            )
            obs.flush()
            self.state.result = data.get("result", "")
            self.state.prompt_versions = data.get("prompt_versions", {})
            self.state.stdout = data.get("stdout", "")
            self.state.stderr = data.get("stderr", "")
            return data
        except Exception:
            log.exception("run_fitness_plan failed")  # ensure traceback appears in server logs
            raise
