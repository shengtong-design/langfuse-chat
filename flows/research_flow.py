from __future__ import annotations

from typing import Any, Callable, Dict

from crewai.flow.flow import Flow, start
from pydantic import BaseModel


class ResearchState(BaseModel):
    question: str = ""
    result: str = ""
    prompt_versions: Dict[str, str] = {}
    stdout: str = ""
    stderr: str = ""


class ResearchFlow(Flow[ResearchState]):
    """Orchestrates the ResearchCrew for a single question.

    Usage:
        flow = ResearchFlow(connectors_factory=_get_connectors, langfuse_client=lf)
        result = flow.kickoff(inputs={"question": "What is AI?"})
    """

    def __init__(
        self,
        connectors_factory: Callable,
        langfuse_client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._connectors_factory = connectors_factory
        self._langfuse_client = langfuse_client

    @start()
    def run_research(self) -> Dict[str, Any]:
        from core.observability.context import EnrichedConnectorManager, make_run_context
        from crews.research_crew import ResearchCrew

        obs = EnrichedConnectorManager(
            self._connectors_factory(),
            make_run_context("researcher"),
        )
        data = ResearchCrew().run(
            {"question": self.state.question},
            obs,
            langfuse_client=self._langfuse_client,
        )
        obs.flush()
        self.state.result = data.get("result", "")
        self.state.prompt_versions = data.get("prompt_versions", {})
        self.state.stdout = data.get("stdout", "")
        self.state.stderr = data.get("stderr", "")
        return data
