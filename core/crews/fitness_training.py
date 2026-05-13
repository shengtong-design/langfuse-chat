from pathlib import Path
from typing import Any, Dict

import yaml

from .base import BaseCrew
from .common import kickoff_crew

_CONFIG = yaml.safe_load((Path(__file__).parent / "fitness_training.yaml").read_text())


class FitnessTrainingCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.fitness_training"

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        from crewai import Agent, Crew, Task
        from core.observability.context.callbacks import get_crew_kwargs

        agents = {
            name: Agent(**spec, verbose=True, allow_delegation=False)
            for name, spec in _CONFIG["agents"].items()
        }
        tasks = [
            Task(
                description=spec["description"].format(**inputs),
                expected_output=spec["expected_output"],
                agent=agents[spec["agent"]],
            )
            for spec in _CONFIG["tasks"].values()
        ]
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
            **get_crew_kwargs(obs),
        )

        with obs.span(
            "crewai.fitness_training", "chain",
            input_data=inputs,
            metadata={"framework": "crewai", "crew": self.crew_name},
        ) as root:
            result, stdout, stderr = kickoff_crew(crew, obs, input_data=inputs)

            sections = _CONFIG.get("output_sections", [])
            tasks_out = result.tasks_output
            if tasks_out and len(tasks_out) == len(sections):
                combined = "\n\n".join(
                    f"{s['title']}\n\n{tasks_out[i].raw}"
                    for i, s in enumerate(sections)
                )
            else:
                combined = str(result)

            root.update(output={"result": combined})
            return {"result": combined, "stdout": stdout, "stderr": stderr}
