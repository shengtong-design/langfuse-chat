from pathlib import Path
from typing import Any, Dict

import yaml

from .base import BaseCrew
from .common import kickoff_crew

_CONFIG = yaml.safe_load((Path(__file__).parent / "researcher.yaml").read_text())


class ResearcherCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.researcher"

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        from crewai import Agent, Crew, Task
        from core.observability.context.callbacks import get_crew_kwargs
        from core.prompts import PromptLoader

        loader = PromptLoader()
        prompts = {}
        agents = {}
        for name, spec in _CONFIG["agents"].items():
            prompt = loader.get(f"researcher_{name}", fallback=spec)
            prompts[name] = prompt
            agents[name] = Agent(**prompt.config, verbose=True, allow_delegation=False)

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

        prompt_meta = {
            f"agent.{name}.prompt_version": p.version
            for name, p in prompts.items()
        }

        with obs.span(
            "crewai.research", "chain",
            input_data=inputs,
            metadata={"framework": "crewai", "crew": self.crew_name, **prompt_meta},
        ) as root:
            result, stdout, stderr = kickoff_crew(crew, obs, input_data=inputs)
            output = str(result)
            root.update(output={"result": output})
            return {"result": output, "stdout": stdout, "stderr": stderr}
