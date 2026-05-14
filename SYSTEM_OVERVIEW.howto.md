# How to (re)generate `SYSTEM_OVERVIEW.md`

This guide captures the methodology used to produce `SYSTEM_OVERVIEW.md` and
`SYSTEM_OVERVIEW.html`, so the next regeneration lands at the same shape,
accuracy, and quality.

---

## 1. When to regenerate

**Do** regenerate when:

- A flow / crew / agent / task is added, removed, or restructured.
- The four-layer versioning model (deployment / flow / crew / per-run prompts) changes.
- A new observability connector or backend is added.
- A core invariant changes (LLM-text allowlist, namespace prefix, sort direction, ...).
- The runtime sequence changes meaningfully (new spans, new callbacks, new merge points).

**Don't** regenerate for:

- Prompt edits in Langfuse — those don't change the doc, they change
  `agents_signature` / `tasks_signature` at runtime.
- Bug fixes that don't change behavior, structure, or invariants.
- Typos / formatting tweaks only.

---

## 2. Source-of-truth files to read first

Before writing any section, re-read the files below. Every doc claim must
trace back to one of them.

### Project shape

- `VERSION` — current app semver (never paste the value into the doc).
- `runtime.txt` — Python version pin.
- `requirements.txt` — framework pins (doc points here, never lists pins inline).
- `crew_app.py` — Streamlit entry point, tab wiring, Datadog init logic.

### Domain core

- `crews/base.py` — the central abstraction. Read **in full** every time. Watch for:
  - `_AGENT_LLM_TEXT_FIELDS`, `_TASK_LLM_TEXT_FIELDS`, `_TASK_RESERVED_KEYS`
  - `_AGENT_PROMPT_NAMESPACE`, `_TASK_PROMPT_NAMESPACE`
  - `_namespaced()` invariant (no dots in bare keys)
  - `_pull_llm_text()` allowlist filter
  - `_load_agents`, `_load_tasks`, `_build_prompt_meta`, `run`
  - `crew_version` ClassVar + bump-rule comment
- `crews/<name>_crew.py` — concrete crew declarations.
- `crews/common.py` — `kickoff_crew` (stdout/stderr capture, error mark).

### Flow layer

- `flows/<name>_flow.py` — Pydantic state + `flow_name`/`flow_version` ClassVars
  + the single `@start` body. Bump rules live in the inline comment.

### Prompts

- `core/prompts/loader.py` — `PromptLoader.get` contract, fallback semantics,
  cache TTL, error scenarios.

### Observability

- `core/observability/base.py` — `BaseConnector`, `SpanHandle`, `ObsManager` protocol.
- `core/observability/__init__.py` — `ConnectorManager`, `MultiSpanHandle`, `for_callbacks`.
- `core/observability/langfuse_connector.py`, `datadog_connector.py`.
- `core/observability/span_limits.py` — truncation constants.
- `core/observability/context/run_context.py` — fields + three projections (as_metadata, as_tags, as_dd_tags).
- `core/observability/context/session.py` — `make_run_context` factory.
- `core/observability/context/enriched.py` — `_merged_metadata` (the *single*
  sort source of truth — `sorted(reverse=True)` couples to Langfuse UI rendering).
- `core/observability/context/callbacks.py` — `CrewCallbacks`,
  `_StepCallback`, `_TaskCallback`.

### Config & CLIs

- `config/environments/*.yaml` — per-env non-secrets (`deployment_sha`, `model_defaults`).
- `scripts/seed_prompts.py` — generates the inventory listed in §5.6.

### YAML inventory (for §5 tables)

- `agents/*.yaml` — verify `prompt_key`, `fallback.role`, optional `tools:` list.
- `tasks/*.yaml` — verify `task_name`, `agent`, `prompt_key`, `fallback`,
  optional `guardrail:` key.

### Tools inventory (for §5.7)

- `tools/__init__.py` — `TOOL_BUILDERS` registry (key → builder(inputs) → BaseTool).
- `tools/*_tool.py` — one `BaseTool` subclass per file; verify `name`,
  `description`, declared instance fields used by the builder.

### Guardrails inventory (for §5.8)

- `guardrails/__init__.py` — `GUARDRAIL_BUILDERS` registry (key →
  builder(inputs) → `(TaskOutput) → (bool, Any)` closure).
- `guardrails/*_guardrail.py` — one builder per file; verify the registry
  key matches the YAML reference and that the inner closure carries no
  return annotation (PEP 563 trap; see `fitness_analysis_guardrail.py`).

---

## 3. The 10-section model

| § | Title | Format |
|---|---|---|
| 1 | Design Tenets | Table — invariants only. |
| 2 | Architecture at a Glance | Mermaid `flowchart TB`. |
| 3 | Project Layout | Mermaid `flowchart LR` (imports) + ASCII file tree. |
| 4 | Core Concepts | Ten subsections, see below. |
| 5 | Component Inventory | Mermaid `flowchart LR` (topology) + six tables (Crews, Agents, Tasks, Langfuse prompts, Tools, Guardrails). |
| 6 | Runtime Sequence | Mermaid `sequenceDiagram` (one worked example). |
| 7 | Configuration | Tables (env vars + env files). |
| 8 | Extension Points | Numbered procedures. |
| 9 | User Instructions | Six subsections, see below. |
| 10 | Related Documents & Pointers | Bullet list. |

### §4 subsections (in order)

1. Flow — user-facing recipe.
2. Crew — inner LLM-driven recipe.
3. Agent — YAML scaffolding + Langfuse-managed text.
4. Task — YAML wiring + Langfuse-managed text.
5. Guardrail — structural validators on task output.
6. PromptLoader — Langfuse with deterministic fallback.
7. Observability — connector layer.
8. Span types, truncation, and CrewAI callbacks.
9. RunContext — the run's identity card.
10. Four-layer versioning model.
11. Trace metadata key order.

### §9 subsections (in order)

1. First-time setup.
2. Run the app.
3. Run an experiment.
4. Read traces in Langfuse.
5. Troubleshooting.
6. Day-to-day workflow.

---

## 4. Hard invariants — non-negotiable

- **No hardcoded version values.** Strip every `1.0.0`, framework pin (`1.14.4`,
  `≥2.11`, ...). Point at `VERSION` / `requirements.txt` / `Crew.crew_version` etc.
  Exception: §8 Extension Points template, where `crew_version = "1.0.0"` is
  the canonical *starting* semver for a new crew (not current state).
- **Doc claims trace back to code.** Every "X happens" claim maps to a
  specific file. Use `file.py:NN` references when the reader might want to
  verify.
- **Concept colors are consistent across diagrams.** See §5 of this guide.
- **Mermaid for diagrams; ASCII only for file trees.** Trees are still the
  best text representation; everything else converts.
- **Each diagram is followed by a "reading the diagram" note** that points
  at the non-obvious bits.

---

## 5. Diagram conventions

### Concept color palette (used across every diagram)

| Concept | Fill | Stroke |
|---|---|---|
| UI | `#ffe0b2` | `#e65100` |
| Flow | `#fff3e0` | `#e65100` |
| Crew | `#e8f5e9` | `#1b5e20` |
| Agent | `#e3f2fd` | `#0d47a1` |
| Task | `#f3e5f5` | `#4a148c` |
| Tool | `#fff8e1` | `#f57f17` |
| Guardrail | `#ffebee` | `#c62828` |
| PromptLoader | `#fce4ec` | `#880e4f` |
| Observability | `#e0f2f1` | `#004d40` |
| External / data store | `#fafafa`, dashed `#616161` |

### Diagram type per section

| Section | Type | Why |
|---|---|---|
| §2 Architecture | `flowchart TB` | Vertical layers map to runtime call stack. |
| §3 Imports | `flowchart LR` | Wide layout fits the package fan-out. |
| §5.1 Topology | `flowchart LR` | Task chain reads left-to-right (sequential order). |
| §6 Runtime | `sequenceDiagram` | Actor-based; `loop` blocks for repeated callbacks; `rect` shading for sub-phases. |

### Diagram content rules

- Node label format: `Name<br/><i>identifier or annotation</i>`.
- Edge labels are short verbs/captions: `kickoff`, `task 1`, `performed by`, `on miss`.
- **Dashed edges** (`-.label.->`) for *file reads* and *performed-by*
  relationships; **solid arrows** for control/data flow.
- Subgraphs for natural groupings (per-pipeline, per-layer).
- For §5.1 specifically: the task chain is the SOLID horizontal path; the
  task-to-agent ownership is a DASHED "performed by" edge; an agent's
  available tools are DASHED "uses" edges from the agent to each Tool
  node. CrewAI defaults to `Process.sequential` and prior task outputs
  auto-flow as implicit context — show this with edge labels like
  `task 2 + ctx of task 1`.

---

## 6. Versioning — what to put where

| Identity | Declared in | Doc treatment |
|---|---|---|
| `app_version` | `VERSION` file | "See `VERSION`" — never paste the value. |
| `flow_version` | `Flow.flow_version` ClassVar | Bump rules in the code comment; doc links to it. |
| `crew_version` | `Crew.crew_version` ClassVar | Same — bump rules in code. |
| `agents_signature` / `tasks_signature` | Computed in `_build_prompt_meta` | Doc shows the *shape* (e.g. `"researcher@2,workout_designer@1"`); no concrete versions. |

The doc carries the **rules** for when to bump and how to interpret. The
**values** stay in code or trace metadata at runtime.

---

## 7. Render to HTML

```bash
py -3.12 scripts/md_to_html.py
```

`scripts/md_to_html.py`:

- Pre-processes ```mermaid fenced blocks into `<div class="mermaid">` raw HTML.
- Injects the `mermaid@11` ESM CDN script + `initialize({ startOnLoad: true })`.
- Emits a self-contained HTML with embedded CSS (sticky table headers,
  row hover, rounded borders, mermaid container styling).
- Requires internet on first load to fetch Mermaid; cached thereafter.

Commit `SYSTEM_OVERVIEW.html` alongside `.md` so readers without a Markdown
viewer can open the HTML directly in a browser.

---

## 8. Verification checklist — run before committing

```bash
# 1. No hardcoded version values (except the §8 template "1.0.0").
grep -n -E "1\.0\.0|≥[0-9]|[0-9]+\.[0-9]+\.[0-9]+" SYSTEM_OVERVIEW.md
# Expect exactly one hit: the line `crew_version = "1.0.0"` in §8 Extension Points.

# 2. Section numbers contiguous.
grep -n -E "^##\s+[0-9]+\.|^###\s+[0-9]+\.[0-9]+" SYSTEM_OVERVIEW.md
# Expect: 1, 2, ..., 10 with subsections 4.1-4.11, 5.1-5.8, 9.1-9.6.

# 3. Cross-references resolve.
grep -n -E "see §?[0-9]+(\.[0-9]+)?" SYSTEM_OVERVIEW.md

# 4. Renderer succeeds.
py -3.12 scripts/md_to_html.py

# 5. Mermaid block count matches diagrams in source.
grep -c 'class="mermaid"' SYSTEM_OVERVIEW.html
# Currently 4 (§2 + §3 + §5.1 + §6).

# 6. Smoke-test imports still work.
py -3.12 -c "import crews.base, crews.research_crew, crews.fitness_crew, \
            flows.research_flow, flows.fitness_flow, \
            core.observability, core.observability.context, core.prompts, \
            tools, guardrails"
```

Then open `SYSTEM_OVERVIEW.html` in a browser and confirm every Mermaid block
renders (no red "Syntax error in text" boxes).

---

## 9. Where new content goes

| New thing | Lands in |
|---|---|
| Design tenet | §1 row. |
| Runtime layer (e.g. a tools registry) | §2 diagram + a new §4 subsection. |
| Top-level directory | §3 imports diagram + file tree. |
| Flow / crew / agent / task / **tool** | §5 inventory (diagram + relevant table). When adding a tool, update §5.1 diagram (Tool node + dashed "uses" edge), §5.3 "Tools wired" column, §5.4 "Tools" column, and §5.7 row. |
| **Guardrail** | §5 inventory (diagram + table). Add a Guardrail node + dashed "guarded by" edge to §5.1, fill the §5.5 "Guardrail" column for the owning task, and add a §5.8 row. Bump the owning `Crew.crew_version`. |
| Core concept | §4 subsection (renumber subsequent if inserted before §4.10). |
| Step in `BaseCrew.run` | §6 sequence diagram. |
| Env var | §7 table. |
| How-to or troubleshooting row | §9 subsection. |
| External doc | §10 bullet. |
| Invariant | §1 tenets + enforce in code; reference the file in §10. |

---

## 10. Reference points

- The 10-section model was established in commit `4bf18c2` (full rewrite).
- Diagrams converted to Mermaid in `65221f5`, `897e68f`.
- §5.1 topology corrected to show task chain + performed-by edges in `f15d438`.
- §5.7 Tools subsection added (with dashed "uses" edges in §5.1, "Tools"
  columns in §5.3 and §5.4) when the `tools/` package shipped.
- The single-source-of-truth metadata sort lives in
  `core/observability/context/enriched.py` (`_merged_metadata`).
- Bump-rule comments live next to the ClassVar declarations they govern, not
  in the doc.
