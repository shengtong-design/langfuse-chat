from typing import Any, Dict

from .base import BaseCrew
from .common import kickoff_crew


class ResearcherCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.researcher"

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        from crewai import Agent, Crew, Task
        from core.observability.context.callbacks import get_crew_kwargs

        question = inputs["question"]

        agent_spec = {
            "role": "Researcher",
            "goal": "Research the user's question and answer clearly and accurately.",
            "backstory": "You are a diligent researcher who writes concise, well-structured answers with examples.",
        }
        task_spec = {
            "description": f'Research the question: "{question}"',
            "expected_output": "A clear, concise answer with key points and 1-3 examples if applicable.",
        }

        researcher = Agent(**agent_spec, verbose=True, allow_delegation=False)
        task = Task(**task_spec, agent=researcher)
        crew = Crew(agents=[researcher], tasks=[task], verbose=True, **get_crew_kwargs(obs))

        with obs.span(
            "crewai.research", "chain",
            input_data={"question": question, "crew": {"agents": [agent_spec], "tasks": [task_spec]}},
            metadata={"framework": "crewai", "crew": self.crew_name},
        ) as root:
            result, stdout, stderr = kickoff_crew(crew, obs, input_data={"question": question})
            output = str(result)
            root.update(output={"result": output})
            return {"result": output, "stdout": stdout, "stderr": stderr}
