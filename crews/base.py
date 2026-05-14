from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List

import yaml
from crewai import Agent, Crew, Task

from core.observability.context.callbacks import get_crew_kwargs
from core.prompts import PromptLoader, PromptResult
from guardrails import GUARDRAIL_BUILDERS
from tools import TOOL_BUILDERS
from .common import kickoff_crew

_AGENTS_DIR = Path(__file__).parent.parent / "agents"
_TASKS_DIR = Path(__file__).parent.parent / "tasks"

# Per-concept LLM-text field sets. Only these keys can be overridden by a
# Langfuse-side edit; everything else stays code/YAML-only so wiring cannot
# drift via Langfuse. Extend per CrewAI's docs when adding new prompt-y fields.
_AGENT_LLM_TEXT_FIELDS = (
    "role", "goal", "backstory",
    "system_template", "prompt_template", "response_template",
)
_TASK_LLM_TEXT_FIELDS = ("description", "expected_output")

# Reserved top-level YAML keys consumed by the loader. Anything outside this set
# is forwarded to Task(**kwargs), so new CrewAI Task fields (context, tools,
# async_execution, output_pydantic, output_file, human_input, callback,
# max_retries, markdown, config, ...) are zero-code to enable.
# NOTE: 'context' and 'output_pydantic' need a resolver pass (name->Task and
# "module:Class" import) before they can flow through raw — add when needed.
# 'guardrail' is resolved against guardrails.GUARDRAIL_BUILDERS below (same
# pattern as tools).
_TASK_RESERVED_KEYS = frozenset({"task_name", "agent", "prompt_key", "fallback"})

# Langfuse prompt namespace per CrewAI concept. The YAML key is kept bare and
# human-readable; the prefix is added at the Langfuse boundary so an agent
# prompt and a task prompt can never collide on the same Langfuse name.
# Extend (don't reuse) for future concepts: crew., tool., knowledge., ...
_AGENT_PROMPT_NAMESPACE = "agent."
_TASK_PROMPT_NAMESPACE = "task."


def _namespaced(namespace: str, key: str, *, source: str) -> str:
    """Prepend the concept namespace to a bare YAML key.

    Raises ValueError if the key already contains a dot — bare-key invariant
    prevents accidental double-namespacing (e.g. "agent.agent.researcher").
    """
    if "." in key:
        raise ValueError(
            f"{source} prompt key {key!r} must be bare (no dots); "
            f"the {namespace!r} prefix is added automatically."
        )
    return f"{namespace}{key}"


def _pull_llm_text(prompt_config: Dict[str, Any], fallback: Dict[str, Any], fields: tuple) -> Dict[str, Any]:
    """Resolve LLM-text fields from Langfuse with fallback, ignoring extras.

    Only keys in ``fields`` are returned. Any other keys Langfuse may return
    are dropped so wiring cannot be overridden via a Langfuse edit.
    """
    out: Dict[str, Any] = {}
    for field in fields:
        if field in prompt_config:
            out[field] = prompt_config[field]
        elif field in fallback:
            out[field] = fallback[field]
    return out


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

    def _load_agents(self, inputs: Dict[str, Any]) -> tuple:
        """Load agent YAMLs, fetch prompts, return (agents_dict, prompts_dict).

        Each agent's ``tools:`` YAML list (optional) is resolved against
        ``tools.TOOL_BUILDERS``; builders receive ``inputs`` so per-run state
        (e.g. uploaded file paths) can flow into tool construction without
        leaking through Langfuse-editable prompt text.
        """
        agent_specs = {
            Path(n).stem: yaml.safe_load((_AGENTS_DIR / n).read_text())
            for n in self._agent_yaml_names
        }
        loader = PromptLoader()
        agents: Dict[str, Agent] = {}
        prompts: Dict[str, PromptResult] = {}
        for name, spec in agent_specs.items():
            prompt_key = spec.get("prompt_key") or name
            langfuse_name = _namespaced(_AGENT_PROMPT_NAMESPACE, prompt_key, source=f"agents/{name}.yaml")
            fallback = spec.get("fallback", {}) or {}
            prompt = loader.get(langfuse_name, fallback=fallback)
            prompts[name] = prompt
            agent_text = _pull_llm_text(prompt.config, fallback, _AGENT_LLM_TEXT_FIELDS)
            tools = []
            for tool_key in spec.get("tools") or []:
                if tool_key not in TOOL_BUILDERS:
                    raise ValueError(
                        f"agent YAML agents/{name}.yaml references unknown tool key "
                        f"{tool_key!r}; register it in tools/__init__.py:TOOL_BUILDERS"
                    )
                tools.append(TOOL_BUILDERS[tool_key](inputs))
            agents[name] = Agent(
                **agent_text,
                tools=tools,
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
            langfuse_name = _namespaced(_TASK_PROMPT_NAMESPACE, prompt_key, source=f"tasks/{filename}")

            prompt = loader.get(langfuse_name, fallback=fallback)
            task_text = _pull_llm_text(prompt.config, fallback, _TASK_LLM_TEXT_FIELDS)
            description = task_text["description"]
            expected_output = task_text.get("expected_output", "")

            wiring_kwargs = {k: v for k, v in spec.items() if k not in _TASK_RESERVED_KEYS}

            guardrail_key = wiring_kwargs.get("guardrail")
            if isinstance(guardrail_key, str):
                if guardrail_key not in GUARDRAIL_BUILDERS:
                    raise ValueError(
                        f"task YAML {filename} references unknown guardrail key "
                        f"{guardrail_key!r}; register it in guardrails/__init__.py:GUARDRAIL_BUILDERS"
                    )
                wiring_kwargs["guardrail"] = GUARDRAIL_BUILDERS[guardrail_key](inputs)

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
        agents, agent_prompts = self._load_agents(inputs)
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
