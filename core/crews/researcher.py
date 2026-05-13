import contextlib
import io
from typing import Any, Dict

from .base import BaseCrew
from core.observability.context.callbacks import get_crew_kwargs


class ResearcherCrew(BaseCrew):
    @property
    def crew_name(self) -> str:
        return "crewai.researcher"

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        from crewai import Agent, Crew, Task

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

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with obs.span(
            "crewai.research",
            "chain",
            input_data={"question": question, "crew": {"agents": [agent_spec], "tasks": [task_spec]}},
            metadata={"framework": "crewai", "crew": self.crew_name},
        ) as root:
            with obs.span(
                "crew.kickoff",
                "span",
                input_data={"question": question, "agents": [agent_spec], "tasks": [task_spec]},
            ) as kickoff:
                try:
                    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                        result = crew.kickoff()
                    output = str(result)
                    kickoff.update(output={"result": output, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()})
                    root.update(output={"result": output})
                    return {"result": output, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}
                except Exception as e:
                    kickoff.update(output={"error": repr(e), "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}, level="ERROR")
                    root.update(output={"error": repr(e)}, level="ERROR")
                    raise
