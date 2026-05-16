from .base import BaseCrew


class ResearchCrew(BaseCrew):
    crew_version = "1.0.0"

    @property
    def crew_name(self) -> str:
        return "crewai.researcher"

    @property
    def _agent_yaml_names(self) -> list[str]:
        return ["researcher.yaml"]

    @property
    def _task_yaml_names(self) -> list[str]:
        return ["research_task.yaml"]
