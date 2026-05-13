from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import yaml
from crewai import Agent, Crew, Task

from core.observability.context.callbacks import get_crew_kwargs
from core.prompts import PromptLoader
from .common import kickoff_crew

_AGENTS_DIR = Path(__file__).parent.parent / "agents"
_TASKS_DIR = Path(__file__).parent.parent / "tasks"


class BaseCrew(ABC):
    """Template-method base for all crews.

    Subclasses must implement:
      - crew_name         → str
      - _agent_yaml_names → list of filenames in agents/ (e.g. ["researcher.yaml"])
      - _task_yaml_names  → list of filenames in tasks/, in execution order

    Subclasses may override:
      - _format_result(crew_result, task_outputs) → str  for custom output formatting
    """

    @property
    @abstractmethod
    def crew_name(self) -> str: ...

    @property
    @abstractmethod
    def _agent_yaml_names(self) -> List[str]: ...

    @property
    @abstractmethod
    def _task_yaml_names(self) -> List[str]: ...

    def _format_result(self, crew_result: Any, task_outputs: List[Any]) -> str:
        return str(crew_result)

    def run(
        self,
        inputs: Dict[str, Any],
        obs: Any,
    ) -> Dict[str, Any]:
        agent_specs = {
            Path(n).stem: yaml.safe_load((_AGENTS_DIR / n).read_text())
            for n in self._agent_yaml_names
        }
        task_specs = [
            yaml.safe_load((_TASKS_DIR / n).read_text())
            for n in self._task_yaml_names
        ]

        loader = PromptLoader()
        agents: Dict[str, Agent] = {}
        prompts = {}
        for name, spec in agent_specs.items():
            prompt_key = spec.get("agent_name") or name
            prompt = loader.get(
                prompt_key,
                fallback=spec.get("fallback", {}),
            )
            prompts[name] = prompt
            agents[name] = Agent(
                **prompt.config,
                verbose=spec.get("verbose", True),
                allow_delegation=spec.get("allow_delegation", False),
            )

        safe_inputs = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in inputs.items()}
        tasks = []
        for spec in task_specs:
            if "description" not in spec or "agent" not in spec:
                raise ValueError(f"task YAML is missing required key(s) 'description' or 'agent': {spec}")
            tasks.append(Task(
                description=spec["description"].format(**safe_inputs),
                expected_output=spec.get("expected_output", ""),
                agent=agents[spec["agent"]],
            ))

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
            **get_crew_kwargs(obs),
        )

        prompt_meta = {}
        for name, p in prompts.items():
            prompt_meta[f"agent.{name}.prompt_name"] = p.name
            prompt_meta[f"agent.{name}.prompt_version"] = p.version
            prompt_meta[f"agent.{name}.prompt_source"] = "langfuse" if p.version != "fallback" else "yaml_fallback"
            for field in ("role", "goal", "backstory"):
                if field in p.config:
                    prompt_meta[f"agent.{name}.{field}"] = p.config[field]

        # crew_version = highest prompt version loaded (reflects actual runtime behaviour)
        live_versions = [int(p.version) for p in prompts.values() if p.version.isdigit()]
        crew_version = str(max(live_versions)) if live_versions else "fallback"
        ctx = getattr(obs, "_ctx", None)
        if ctx is not None:
            obs._ctx = ctx.with_crew_version(crew_version)

        with obs.span(
            self.crew_name, "chain",
            input_data=inputs,
            metadata={"framework": "crewai", "crew": self.crew_name, **prompt_meta},
        ) as root:
            result, stdout, stderr = kickoff_crew(crew, obs, input_data=inputs)
            output = self._format_result(result, result.tasks_output or [])
            root.update(output={"result": output})
            return {
                "result": output,
                "stdout": stdout,
                "stderr": stderr,
                "prompt_versions": prompt_meta,
            }
