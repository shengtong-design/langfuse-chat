"""Static introspection of a CrewAI Flow → Crew → Agents → Tasks graph.

Used by `evalops.reporter` to render Section 4 (Flow architecture)
of the eval report — a Mermaid diagram + structured text describing
exactly what was kicked off, and what the input/output surface looks
like.

Inputs assumed:
- Flow class: subclass of `crewai.flow.flow.Flow[StateModel]`. We pull
  flow_name + flow_version from class attrs and the State model from
  the `Flow[State]` generic argument.
- Inner Crew class: located by static hint table keyed on Flow class
  name (extend `_FLOW_TO_CREW_HINTS` when registering new flows).
- Crew class: subclass of `crews.base.BaseCrew` with
  `_agent_yaml_names` / `_task_yaml_names` props pointing at files
  under the project's `agents/` and `tasks/` dirs.
- Agent YAML: `prompt_key`, `fallback.role`.
- Task YAML: `task_name`, `agent`, `prompt_key`, `fallback.description`.

All file reads + crew imports happen inside `introspect_flow()` so
this module is import-safe (no transitive flows/ crews/ pulls at
import time).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_DIR = _PROJECT_ROOT / "agents"
_TASKS_DIR = _PROJECT_ROOT / "tasks"

# Map Flow class name → (crew_module, crew_class). Extend as new flows land.
_FLOW_TO_CREW_HINTS: dict[str, tuple[str, str]] = {
    "ResearchFlow": ("crews.research_crew", "ResearchCrew"),
    "FitnessFlow": ("crews.fitness_crew", "FitnessCrew"),
}


@dataclass(frozen=True)
class AgentInfo:
    name: str
    yaml_file: str
    prompt_key: str
    fallback_role: str
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskInfo:
    name: str
    yaml_file: str
    agent: str
    prompt_key: str
    fallback_description: str
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    guardrail: str = ""


@dataclass(frozen=True)
class CrewInfo:
    class_name: str
    crew_name: str
    crew_version: str
    agents: tuple[AgentInfo, ...]
    tasks: tuple[TaskInfo, ...]


@dataclass(frozen=True)
class StateField:
    name: str
    type_name: str
    default: Any


@dataclass(frozen=True)
class FlowGraph:
    class_name: str
    flow_name: str
    flow_version: str
    state_class: str | None
    state_fields: tuple[StateField, ...]
    inputs: tuple[str, ...]   # state fields set by kickoff
    outputs: tuple[str, ...]  # state fields populated by the flow run
    crew: CrewInfo | None
    notes: tuple[str, ...] = field(default_factory=tuple)


def introspect_flow(flow_cls: type) -> FlowGraph:
    """Build a FlowGraph from a Flow class. Best-effort — never raises."""
    notes: list[str] = []

    class_name = flow_cls.__name__
    flow_name = getattr(flow_cls, "flow_name", class_name.lower())
    flow_version = str(getattr(flow_cls, "flow_version", "?"))

    state_cls = _resolve_state_class(flow_cls)
    state_fields = _collect_state_fields(state_cls) if state_cls else ()
    inputs, outputs = _classify_state_fields(flow_cls, state_fields)

    crew_info: CrewInfo | None = None
    hint = _FLOW_TO_CREW_HINTS.get(class_name)
    if hint is None:
        notes.append(f"No crew hint registered for {class_name}; crew section omitted.")
    else:
        crew_module, crew_class = hint
        try:
            mod = importlib.import_module(crew_module)
            crew_cls = getattr(mod, crew_class)
            crew_info = _introspect_crew(crew_cls)
        except Exception as e:
            notes.append(f"Crew introspection failed for {crew_class}: {e!r}")

    return FlowGraph(
        class_name=class_name,
        flow_name=str(flow_name),
        flow_version=flow_version,
        state_class=(state_cls.__name__ if state_cls else None),
        state_fields=state_fields,
        inputs=inputs,
        outputs=outputs,
        crew=crew_info,
        notes=tuple(notes),
    )


def _resolve_state_class(flow_cls: type) -> type | None:
    for base in getattr(flow_cls, "__orig_bases__", ()):
        args = getattr(base, "__args__", None)
        if args:
            return args[0]
    return None


def _collect_state_fields(state_cls: type) -> tuple[StateField, ...]:
    model_fields = getattr(state_cls, "model_fields", None)
    if not model_fields:
        return ()
    out: list[StateField] = []
    for name, info in model_fields.items():
        annotation = getattr(info, "annotation", None)
        type_name = _type_name(annotation)
        default = _safe_default(info)
        out.append(StateField(name=name, type_name=type_name, default=default))
    return tuple(out)


def _classify_state_fields(
    flow_cls: type,
    state_fields: tuple[StateField, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Inputs = fields likely set by kickoff. Outputs = fields set by the run.

    Heuristic: inspect the Flow class source for `self.state.X = ...`
    assignments. Those are outputs. Everything else with a benign default
    is treated as an input candidate.
    """
    if not state_fields:
        return ((), ())
    import inspect
    try:
        src = inspect.getsource(flow_cls)
    except (OSError, TypeError):
        return (tuple(f.name for f in state_fields), ())

    outputs: list[str] = []
    inputs: list[str] = []
    for f in state_fields:
        marker = f"self.state.{f.name}"
        if f"{marker} =" in src or f"{marker}=" in src.replace(" ", ""):
            outputs.append(f.name)
        else:
            inputs.append(f.name)
    return tuple(inputs), tuple(outputs)


def _introspect_crew(crew_cls: type) -> CrewInfo:
    inst = crew_cls()
    agent_yaml_names: list[str] = list(inst._agent_yaml_names)
    task_yaml_names: list[str] = list(inst._task_yaml_names)

    agents = tuple(_load_agent(_AGENTS_DIR / name) for name in agent_yaml_names)
    tasks = tuple(_load_task(_TASKS_DIR / name) for name in task_yaml_names)

    return CrewInfo(
        class_name=crew_cls.__name__,
        crew_name=str(inst.crew_name),
        crew_version=str(getattr(crew_cls, "crew_version", "?")),
        agents=agents,
        tasks=tasks,
    )


def _load_agent(path: Path) -> AgentInfo:
    data = _safe_load_yaml(path)
    fallback = data.get("fallback") or {}
    return AgentInfo(
        name=path.stem,
        yaml_file=path.name,
        prompt_key=str(data.get("prompt_key") or path.stem),
        fallback_role=str(fallback.get("role") or ""),
        tools=_as_str_tuple(data.get("tools")),
        skills=_as_str_tuple(data.get("skills")),
    )


def _load_task(path: Path) -> TaskInfo:
    data = _safe_load_yaml(path)
    fallback = data.get("fallback") or {}
    guardrail = data.get("guardrail")
    return TaskInfo(
        name=str(data.get("task_name") or path.stem),
        yaml_file=path.name,
        agent=str(data.get("agent") or ""),
        prompt_key=str(data.get("prompt_key") or path.stem),
        fallback_description=str(fallback.get("description") or ""),
        tools=_as_str_tuple(data.get("tools")),
        skills=_as_str_tuple(data.get("skills")),
        guardrail=str(guardrail) if guardrail else "",
    )


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v) for v in value if v)
    return (str(value),)


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _type_name(annotation: Any) -> str:
    if annotation is None:
        return "Any"
    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")


def _safe_default(field_info: Any) -> Any:
    try:
        d = field_info.default
    except Exception:
        return None
    # Pydantic sentinel for "no default"
    name = type(d).__name__
    if name == "PydanticUndefinedType":
        return "<required>"
    return d


# ─────────────────────────── rendering ──────────────────────────────────────


def render_mermaid(graph: FlowGraph) -> str:
    """Return a Mermaid flowchart string for Flow → Crew → Agents/Tasks → Tools/Skills/Guardrails."""
    lines = ["flowchart TD"]

    input_node_label = (
        "<br/>".join(f"{n}: {_field_type(graph, n)}" for n in graph.inputs)
        or "(no inputs)"
    )
    output_node_label = (
        "<br/>".join(f"{n}: {_field_type(graph, n)}" for n in graph.outputs)
        or "(no outputs)"
    )

    lines.append(f'    IN["Kickoff inputs<br/>{input_node_label}"]')
    lines.append(
        f'    FLOW["{graph.class_name}<br/>flow_version={graph.flow_version}"]'
    )
    if graph.crew:
        lines.append(
            f'    CREW["{graph.crew.class_name}<br/>'
            f'crew_version={graph.crew.crew_version}"]'
        )
        for i, a in enumerate(graph.crew.agents):
            lines.append(f'    A{i}["Agent: {a.name}<br/>prompt_key={a.prompt_key}"]')
        for j, t in enumerate(graph.crew.tasks):
            lines.append(
                f'    T{j}["Task: {t.name}<br/>'
                f'prompt_key={t.prompt_key}<br/>agent={t.agent}"]'
            )

        # Wiring layers: tools, skills, guardrails — uniquely keyed across agents/tasks.
        wiring_ids: dict[tuple[str, str], str] = {}
        next_id = 0

        def _wid(kind: str, name: str) -> str:
            nonlocal next_id
            key = (kind, name)
            if key not in wiring_ids:
                wiring_ids[key] = f"{kind[:1].upper()}{next_id}"
                next_id += 1
            return wiring_ids[key]

        # Declare wiring nodes first
        for a in graph.crew.agents:
            for name in a.tools:
                _wid("tool", name)
            for name in a.skills:
                _wid("skill", name)
        for t in graph.crew.tasks:
            for name in t.tools:
                _wid("tool", name)
            for name in t.skills:
                _wid("skill", name)
            if t.guardrail:
                _wid("guardrail", t.guardrail)
        for (kind, name), nid in wiring_ids.items():
            label_prefix = {"tool": "Tool", "skill": "Skill", "guardrail": "Guardrail"}[kind]
            lines.append(f'    {nid}(["{label_prefix}: {name}"])')

    lines.append(f'    OUT["Flow outputs<br/>{output_node_label}"]')

    lines.append("    IN -->|kickoff| FLOW")
    if graph.crew:
        lines.append("    FLOW -->|Crew.run| CREW")
        for i, a in enumerate(graph.crew.agents):
            lines.append(f"    CREW --> A{i}")
        for j, t in enumerate(graph.crew.tasks):
            lines.append(f"    CREW --> T{j}")
            for i, a in enumerate(graph.crew.agents):
                if a.name == t.agent:
                    lines.append(f"    A{i} -.->|executes| T{j}")

        # Edges from agents/tasks to wiring nodes
        for i, a in enumerate(graph.crew.agents):
            for name in a.tools:
                lines.append(f"    A{i} -.->|uses| {wiring_ids[('tool', name)]}")
            for name in a.skills:
                lines.append(f"    A{i} -.->|uses| {wiring_ids[('skill', name)]}")
        for j, t in enumerate(graph.crew.tasks):
            for name in t.tools:
                lines.append(f"    T{j} -.->|uses| {wiring_ids[('tool', name)]}")
            for name in t.skills:
                lines.append(f"    T{j} -.->|uses| {wiring_ids[('skill', name)]}")
            if t.guardrail:
                lines.append(
                    f"    T{j} -.->|gated by| {wiring_ids[('guardrail', t.guardrail)]}"
                )

        lines.append("    CREW --> OUT")
    else:
        lines.append("    FLOW --> OUT")

    return "\n".join(lines)


def render_text_tree(graph: FlowGraph) -> str:
    """Return a plaintext tree for environments where Mermaid won't render (PDF)."""
    out: list[str] = []
    out.append(f"Flow: {graph.class_name} (flow_version={graph.flow_version})")
    if graph.inputs:
        out.append("├── Kickoff inputs:")
        for n in graph.inputs:
            out.append(f"│     • {n}: {_field_type(graph, n)}")
    if graph.crew:
        out.append("├── Crew: " + graph.crew.class_name +
                   f" (crew_version={graph.crew.crew_version}, span={graph.crew.crew_name})")
        if graph.crew.agents:
            out.append("│   ├── Agents:")
            for a in graph.crew.agents:
                role = f" — {a.fallback_role}" if a.fallback_role else ""
                out.append(f"│   │     • {a.name} (prompt_key={a.prompt_key}){role}")
                if a.tools:
                    out.append(f"│   │         tools:  {', '.join(a.tools)}")
                if a.skills:
                    out.append(f"│   │         skills: {', '.join(a.skills)}")
        if graph.crew.tasks:
            out.append("│   └── Tasks:")
            for t in graph.crew.tasks:
                out.append(
                    f"│         • {t.name} → agent={t.agent}, "
                    f"prompt_key={t.prompt_key}"
                )
                if t.tools:
                    out.append(f"│             tools:     {', '.join(t.tools)}")
                if t.skills:
                    out.append(f"│             skills:    {', '.join(t.skills)}")
                if t.guardrail:
                    out.append(f"│             guardrail: {t.guardrail}")
    if graph.outputs:
        out.append("└── Flow outputs:")
        for n in graph.outputs:
            out.append(f"      • {n}: {_field_type(graph, n)}")
    return "\n".join(out)


def _field_type(graph: FlowGraph, name: str) -> str:
    for f in graph.state_fields:
        if f.name == name:
            return f.type_name
    return "?"
