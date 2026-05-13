from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List

import yaml
from crewai import Agent, Crew, Task

from core.observability.context.callbacks import get_crew_kwargs
from core.prompts import PromptLoader
from .common import kickoff_crew

_AGENTS_DIR = Path(__file__).parent.parent / "agents"
_TASKS_DIR = Path(__file__).parent.parent / "tasks"


class BaseCrew(ABC):
    """Template-method base for all crews.

    Subclasses must set/implement:
      - crew_version      → semver string identifying the crew recipe
      - crew_name         → str
      - _agent_yaml_names → list of filenames in agents/ (e.g. ["researcher.yaml"])
      - _task_yaml_names  → list of filenames in tasks/, in execution order

    Subclasses may override:
      - _format_result(crew_result, task_outputs) → str  for custom output formatting
    """

    # Recipe version owned by the crew definition. Semver string, bumped manually
    # via PR. Distinct from per-run prompt versions (recorded as agents_signature).
    #
    # Bump when:
    #   - _agent_yaml_names changes (add/remove/swap an agent)
    #   - _task_yaml_names changes (reorder, add, or remove a task)
    #   - _format_result semantics change
    #   - the wired tool set changes
    #   - an agent's prompt key is renamed (different Langfuse prompt resolves)
    # Do NOT bump for a Langfuse-side edit to an existing agent's prompt — that
    # is already captured per-run in agents_signature.
    crew_version: ClassVar[str] = ""

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_agents(self) -> tuple:
        """Load agent YAMLs, fetch prompts, return (agents_dict, prompts_dict)."""
        agent_specs = {
            Path(n).stem: yaml.safe_load((_AGENTS_DIR / n).read_text())
            for n in self._agent_yaml_names
        }
        loader = PromptLoader()
        agents: Dict[str, Agent] = {}
        prompts = {}
        for name, spec in agent_specs.items():
            prompt_key = spec.get("agent_name") or name
            prompt = loader.get(prompt_key, fallback=spec.get("fallback", {}))
            prompts[name] = prompt
            agents[name] = Agent(
                **prompt.config,
                verbose=spec.get("verbose", True),
                allow_delegation=spec.get("allow_delegation", False),
            )
        return agents, prompts

    def _load_tasks(self, agents: Dict[str, Agent], inputs: Dict[str, Any]) -> List[Task]:
        """Load task YAMLs and build Task objects bound to loaded agents."""
        safe_inputs = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in inputs.items()}
        tasks = []
        for n in self._task_yaml_names:
            spec = yaml.safe_load((_TASKS_DIR / n).read_text())
            if "description" not in spec or "agent" not in spec:
                raise ValueError(f"task YAML missing required key(s) 'description' or 'agent': {spec}")
            tasks.append(Task(
                description=spec["description"].format(**safe_inputs),
                expected_output=spec.get("expected_output", ""),
                agent=agents[spec["agent"]],
            ))
        return tasks

    def _build_prompt_meta(self, prompts: dict) -> Dict[str, Any]:
        """Build per-agent prompt metadata + an agents_signature for trace filtering.

        agents_signature is a deterministic, sorted, lossless record of which prompt
        version resolved for each agent on this run (e.g. "researcher@6,summarizer@3").
        Distinct from crew_version, which identifies the recipe and is set on RunContext.
        """
        prompt_meta: Dict[str, Any] = {}
        for name, p in prompts.items():
            prompt_meta[f"agent.{name}.prompt_name"] = p.name
            prompt_meta[f"agent.{name}.prompt_version"] = p.version
            prompt_meta[f"agent.{name}.prompt_source"] = "langfuse" if p.version != "fallback" else "yaml_fallback"
            for field in ("role", "goal", "backstory"):
                if field in p.config:
                    prompt_meta[f"agent.{name}.{field}"] = p.config[field]
        prompt_meta["agents_signature"] = ",".join(
            f"{name}@{p.version}" for name, p in sorted(prompts.items())
        )
        return prompt_meta

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        assert self.crew_version, f"{type(self).__name__}.crew_version must be set"
        agents, prompts = self._load_agents()
        tasks = self._load_tasks(agents, inputs)
        crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=True, **get_crew_kwargs(obs))
        prompt_meta = self._build_prompt_meta(prompts)
        with obs.span(
            self.crew_name, "chain",
            input_data=inputs,
            metadata={"framework": "crewai", "crew": self.crew_name, **prompt_meta},
        ) as root:
            result, stdout, stderr = kickoff_crew(crew, obs, input_data=inputs)
            output = self._format_result(result, result.tasks_output or [])
            root.set_output({"result": output})
            return {
                "result": output,
                "stdout": stdout,
                "stderr": stderr,
                "prompt_versions": prompt_meta,
            }
