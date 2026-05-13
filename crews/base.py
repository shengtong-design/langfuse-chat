from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List

import yaml
from crewai import Agent, Crew, Task

from core.observability.context.callbacks import get_crew_kwargs
from core.prompts import PromptLoader, PromptResult
from .common import kickoff_crew

_AGENTS_DIR = Path(__file__).parent.parent / "agents"
_TASKS_DIR = Path(__file__).parent.parent / "tasks"

# Per-concept schema for the "task" YAML. LLM-text fields are the only keys a
# Langfuse-side edit can override; everything else stays code/YAML-only so wiring
# (agent, context, tools, output schema, retries, ...) cannot drift via Langfuse.
_TASK_LLM_TEXT_FIELDS = ("description", "expected_output")
# Reserved top-level YAML keys consumed by the loader. Anything outside this set
# is forwarded to Task(**kwargs), so new CrewAI Task fields (context, tools,
# async_execution, output_pydantic, output_file, guardrail, human_input,
# callback, max_retries, markdown, config, ...) are zero-code to enable.
# NOTE: 'context' and 'output_pydantic' need a resolver pass (name->Task and
# "module:Class" import) before they can flow through raw — add when needed.
_TASK_RESERVED_KEYS = frozenset({"task_name", "agent", "prompt_key", "fallback"})


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
    # via PR. Distinct from per-run prompt versions (recorded as agents_signature
    # and tasks_signature).
    #
    # Bump when:
    #   - _agent_yaml_names changes (add/remove/swap an agent)
    #   - _task_yaml_names changes (reorder, add, or remove a task)
    #   - _format_result semantics change
    #   - the wired tool set changes
    #   - an agent's or task's prompt key is renamed (different Langfuse prompt resolves)
    #   - task-YAML wiring changes (agent assignment, context, tools, output schema, ...)
    # Do NOT bump for a Langfuse-side edit to an existing agent's or task's
    # prompt — that is already captured per-run in agents_signature/tasks_signature.
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
        prompts: Dict[str, PromptResult] = {}
        for name, spec in agent_specs.items():
            prompt_key = spec.get("agent_name") or name
            prompt = loader.get(prompt_key, fallback=spec.get("fallback", {}))
            prompts[name] = prompt
            # NOTE: this merge is permissive — any key Langfuse returns in
            # prompt.config flows into Agent(...). Tasks use a tighter pull
            # (only the LLM-text fields). Consider mirroring that hardening
            # here so a Langfuse edit cannot silently change agent wiring.
            agents[name] = Agent(
                **prompt.config,
                verbose=spec.get("verbose", True),
                allow_delegation=spec.get("allow_delegation", False),
            )
        return agents, prompts

    def _load_tasks(self, agents: Dict[str, Agent], inputs: Dict[str, Any]) -> tuple:
        """Load task YAMLs and build Task objects bound to loaded agents.

        Returns (tasks_list, task_prompts_dict). Task description and
        expected_output resolve from Langfuse with YAML fallback; all other YAML
        keys are forwarded to Task(**kwargs) as wiring.
        """
        safe_inputs = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in inputs.items()}
        loader = PromptLoader()
        tasks: List[Task] = []
        task_prompts: Dict[str, PromptResult] = {}
        for filename in self._task_yaml_names:
            spec = yaml.safe_load((_TASKS_DIR / filename).read_text())
            if "agent" not in spec or "fallback" not in spec:
                raise ValueError(
                    f"task YAML {filename} missing required key(s) 'agent' or 'fallback'"
                )
            fallback = spec["fallback"] or {}
            if "description" not in fallback:
                raise ValueError(
                    f"task YAML {filename} fallback missing required key 'description'"
                )

            stem = Path(filename).stem
            task_name = spec.get("task_name") or stem
            prompt_key = spec.get("prompt_key") or stem

            prompt = loader.get(prompt_key, fallback=fallback)
            # Pull only the LLM-text fields; ignore anything else Langfuse
            # might return so wiring cannot be overridden via a Langfuse edit.
            description = prompt.config.get("description", fallback["description"])
            expected_output = prompt.config.get(
                "expected_output", fallback.get("expected_output", "")
            )

            wiring_kwargs = {k: v for k, v in spec.items() if k not in _TASK_RESERVED_KEYS}

            tasks.append(Task(
                description=description.format(**safe_inputs),
                expected_output=expected_output.format(**safe_inputs) if expected_output else "",
                agent=agents[spec["agent"]],
                **wiring_kwargs,
            ))
            task_prompts[task_name] = prompt
        return tasks, task_prompts

    def _build_prompt_meta(
        self,
        agent_prompts: Dict[str, PromptResult],
        task_prompts: Dict[str, PromptResult],
    ) -> Dict[str, Any]:
        """Build per-agent and per-task prompt metadata + signatures for trace filtering.

        agents_signature and tasks_signature are deterministic, sorted, lossless
        records of which prompt version resolved on this run (e.g.
        "researcher@6,summarizer@3"). Distinct from crew_version, which
        identifies the recipe and is set on RunContext.
        """
        prompt_meta: Dict[str, Any] = {}
        for name, p in agent_prompts.items():
            prompt_meta[f"agent.{name}.prompt_name"] = p.name
            prompt_meta[f"agent.{name}.prompt_version"] = p.version
            prompt_meta[f"agent.{name}.prompt_source"] = "langfuse" if p.version != "fallback" else "yaml_fallback"
            for field in ("role", "goal", "backstory"):
                if field in p.config:
                    prompt_meta[f"agent.{name}.{field}"] = p.config[field]
        prompt_meta["agents_signature"] = ",".join(
            f"{name}@{p.version}" for name, p in sorted(agent_prompts.items())
        )
        for name, p in task_prompts.items():
            prompt_meta[f"task.{name}.prompt_name"] = p.name
            prompt_meta[f"task.{name}.prompt_version"] = p.version
            prompt_meta[f"task.{name}.prompt_source"] = "langfuse" if p.version != "fallback" else "yaml_fallback"
        prompt_meta["tasks_signature"] = ",".join(
            f"{name}@{p.version}" for name, p in sorted(task_prompts.items())
        )
        return prompt_meta

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        assert self.crew_version, f"{type(self).__name__}.crew_version must be set"
        agents, agent_prompts = self._load_agents()
        tasks, task_prompts = self._load_tasks(agents, inputs)
        crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=True, **get_crew_kwargs(obs))
        prompt_meta = self._build_prompt_meta(agent_prompts, task_prompts)
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
