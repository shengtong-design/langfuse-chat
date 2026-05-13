from typing import Any, List

from .base import BaseCrew

_OUTPUT_SECTIONS = [
    {"title": "## Fitness Profile Analysis", "task": "analysis"},
    {"title": "## Workout Program",          "task": "workout"},
    {"title": "## Nutrition Plan",           "task": "nutrition"},
]


class FitnessCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.fitness_training"

    @property
    def _agent_yaml_names(self) -> List[str]:
        return [
            "fitness_analyst.yaml",
            "workout_designer.yaml",
            "nutrition_advisor.yaml",
        ]

    @property
    def _task_yaml_names(self) -> List[str]:
        return [
            "fitness_analysis_task.yaml",
            "fitness_workout_task.yaml",
            "fitness_nutrition_task.yaml",
        ]

    def _format_result(self, crew_result: Any, task_outputs: List[Any]) -> str:
        if task_outputs and len(task_outputs) == len(_OUTPUT_SECTIONS):
            return "\n\n".join(
                f"{s['title']}\n\n{task_outputs[i].raw}"
                for i, s in enumerate(_OUTPUT_SECTIONS)
            )
        return str(crew_result)
