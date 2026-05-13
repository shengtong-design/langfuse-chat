from typing import List

from .base import BaseCrew


class ResearchCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.researcher"

    @property
    def _agent_yaml_names(self) -> List[str]:
        return ["researcher.yaml"]

    @property
    def _task_yaml_names(self) -> List[str]:
        return ["research_task.yaml"]
