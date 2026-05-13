from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar, Dict

from crewai.flow.flow import Flow, start
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ResearchState(BaseModel):
    question: str = ""
    result: str = ""
    prompt_versions: Dict[str, str] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


class ResearchFlow(Flow[ResearchState]):
    """Orchestrates the ResearchCrew for a single question.

    Usage:
        flow = ResearchFlow(connectors_factory=_get_connectors)
        result = flow.kickoff(inputs={"question": "What is AI?"})
    """

    # Flow recipe semver. Bump when the flow body changes: topology
    # (@start/@listen/@router edits), state-model fields, which crew(s) it
    # orchestrates, or post-processing. Independent of ResearchCrew.crew_version.
    flow_version: ClassVar[str] = "1.0.0"
    flow_name: ClassVar[str] = "researcher"

    def __init__(
        self,
        connectors_factory: Callable,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._connectors_factory = connectors_factory

    @start()
    def run_research(self) -> Dict[str, Any]:
        try:
            from core.observability.context import EnrichedConnectorManager, make_run_context
            from crews.research_crew import ResearchCrew

            obs = EnrichedConnectorManager(
                self._connectors_factory(),
                make_run_context(
                    crew_name="researcher",
                    crew_version=ResearchCrew.crew_version,
                    flow_name=self.flow_name,
                    flow_version=self.flow_version,
                ),
            )
            data = ResearchCrew().run(
                {"question": self.state.question},
                obs,
            )
            obs.flush()
            self.state.result = data.get("result", "")
            self.state.prompt_versions = data.get("prompt_versions", {})
            self.state.stdout = data.get("stdout", "")
            self.state.stderr = data.get("stderr", "")
            return data
        except Exception:
            log.exception("run_research failed")  # ensure traceback appears in server logs
            raise
