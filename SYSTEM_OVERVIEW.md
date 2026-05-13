# System Overview — `langfuse-chat`

A multi-crew CrewAI runner with first-class observability (Langfuse + Datadog
LLM Observability), runtime-managed prompts, and an experiment harness. Exposed
through a Streamlit UI; orchestrated through CrewAI Flows.

- **Runtime:** Python 3.12 (`runtime.txt`)
- **Version:** see `VERSION` (currently `1.0.0`)
- **Top-level entry point:** `crew_app.py` (Streamlit)
- **Frameworks:** CrewAI 1.14.4, Streamlit ≥1.30, Langfuse ≥4.0, Pydantic ≥2.11
- **Optional integrations:** Datadog LLM Observability (`ddtrace` ≥2.0)

---

## 1. Goals & Design Tenets

| Tenet | Realization |
|---|---|
| **Separate fast-moving prompt edits from code** | Both agent **and task** prompts live in Langfuse (production label); the repo carries only YAML fallback config for offline / disaster scenarios. |
| **Observability is a first-class connector layer, not an afterthought** | All crew/agent/task spans flow through `ConnectorManager`; backends (Langfuse, Datadog) plug in via `BaseConnector`. |
| **Run identity is structured and propagated** | A `RunContext` (session, run, user, env, app version, crew version, model version, deployment SHA) attaches to every span and tag automatically. |
| **Crew identity ≠ prompt identity** | `crew_version` (recipe semver), `agents_signature`, and `tasks_signature` (per-run resolved prompt versions) are independent axes, all emitted on the trace. |
| **Langfuse names cannot collide across concepts** | Prompts are stored under namespaced names: `agent.<name>`, `task.<name>`. Prefix is enforced by the per-concept loader; YAML keys stay bare. |
| **Wiring stays in code/YAML; only LLM-text is editable in Langfuse** | Loaders pull a fixed allow-list of fields from Langfuse (`role`/`goal`/`backstory` for agents, `description`/`expected_output` for tasks). Anything else Langfuse returns is dropped — a Langfuse edit cannot change wiring (model, tools, agent assignment, retries, ...). |
| **Extensibility by convention, not framework lock-in** | New crews = add YAMLs + subclass `BaseCrew`. New connectors = subclass `BaseConnector`. New CrewAI Task fields = YAML edit only (generic pass-through). |

---

## 2. Architecture at a Glance

```
                    ┌──────────────────────────┐
                    │  Streamlit UI (crew_app) │
                    │  Research │ Fitness │ Experiments
                    └────────────┬─────────────┘
                                 │ inputs
                    ┌────────────▼─────────────┐
                    │       CrewAI Flow         │   flows/<name>_flow.py
                    │  (state + kickoff)        │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │       Crew (BaseCrew)     │   crews/<name>_crew.py
                    │   load agents → tasks →   │
                    │   kickoff → format result │
                    └─┬───────────┬─────────────┘
                      │           │
        ┌─────────────▼──┐   ┌────▼───────────────────┐
        │  PromptLoader  │   │ EnrichedConnectorMgr   │
        │  (Langfuse +   │   │  · ConnectorManager     │
        │   YAML fallback│   │  · RunContext metadata  │
        └────────┬───────┘   │  · CrewAI step/task cbs │
                 │           └────┬──────────────┬─────┘
        Langfuse │                │              │
       agent.*+  │           Langfuse        Datadog
       task.*    │           Connector       Connector
       prompts   │
        ┌────────▼──────────┐ ┌───────────┐  ┌────────────┐
        │  agents/*.yaml +  │ │ Langfuse  │  │ ddtrace    │
        │  tasks/*.yaml     │ │ traces +  │  │ LLMObs     │
        │  (fallback only)  │ │ prompts   │  │ traces     │
        └───────────────────┘ └───────────┘  └────────────┘
```

---

## 3. Project Layout

```
langfuse-chat/
├── crew_app.py                  ← Streamlit entry point (UI, tabs, flow wiring)
├── VERSION                      ← App semver, exposed as RunContext.app_version
├── runtime.txt                  ← Python 3.12 (deployment hint)
├── requirements.txt
├── SYSTEM_OVERVIEW.md           ← (this file)
│
├── flows/                       ← CrewAI Flow entry points (per crew)
│   ├── research_flow.py
│   └── fitness_flow.py
│
├── crews/                       ← Crew implementations (template-method pattern)
│   ├── base.py                  ← BaseCrew: load YAML, fetch prompts, build spans
│   ├── common.py                ← kickoff_crew helper (stdout capture, error mapping)
│   ├── research_crew.py
│   └── fitness_crew.py
│
├── agents/                      ← Agent YAML config (metadata + YAML fallback only)
│   ├── researcher.yaml
│   ├── fitness_analyst.yaml
│   ├── workout_designer.yaml
│   └── nutrition_advisor.yaml
│
├── tasks/                       ← Task YAML (wiring + fallback block; description
│   ├── research_task.yaml         and expected_output resolve from Langfuse at runtime)
│   ├── fitness_analysis_task.yaml
│   ├── fitness_workout_task.yaml
│   └── fitness_nutrition_task.yaml
│
├── core/
│   ├── prompts/
│   │   └── loader.py            ← PromptLoader (Langfuse get_prompt + fallback merge)
│   │
│   └── observability/
│       ├── base.py              ← BaseConnector, ObsManager Protocol, SpanHandle ABC
│       ├── __init__.py          ← ConnectorManager (fan-out)
│       ├── span_limits.py
│       ├── langfuse_connector.py
│       ├── datadog_connector.py
│       └── context/
│           ├── run_context.py   ← RunContext dataclass (session/run/user/env/...)
│           ├── session.py       ← make_run_context() factory
│           ├── enriched.py      ← EnrichedConnectorManager (auto-metadata)
│           └── callbacks.py     ← get_crew_kwargs() — CrewAI step/task callbacks
│
├── config/
│   └── environments/
│       ├── dev.yaml
│       ├── staging.yaml
│       └── prod.yaml            ← deployment_sha, model_defaults
│
├── scripts/                     ← One-off tooling (not part of the app runtime)
│   ├── bootstrap.py
│   ├── seed_prompts.py
│   └── run_experiment.py
│
├── .agents/                     ← Skill packs for AI coding agents (gitignored, see README)
├── .claude/                     ← Claude Code-specific config / skill junctions
├── skills-lock.json             ← Tracked: pins skill packs by source+hash
└── .env                         ← Local secrets (gitignored, NEVER commit)
```

---

## 4. Core Concepts

### 4.1 Crew (recipe)

A **Crew** is a *recipe*: a fixed list of agents (with prompt keys), a fixed
ordered list of tasks, optional result formatting, and a stable `crew_version`
string identifying that recipe.

`BaseCrew` (in `crews/base.py`) is a template-method base. Subclasses declare:

| Attribute / method | Required | Purpose |
|---|---|---|
| `crew_version: ClassVar[str]` | **yes** | Recipe semver — bumped via PR when the recipe changes. |
| `crew_name` (property) | yes | Short identifier used as the root span name (e.g. `crewai.researcher`). |
| `_agent_yaml_names` (property) | yes | Filenames in `agents/`, e.g. `["researcher.yaml"]`. |
| `_task_yaml_names` (property) | yes | Filenames in `tasks/`, in execution order. |
| `_format_result(crew_result, task_outputs)` | optional | Custom output formatting (defaults to `str(crew_result)`). |

`BaseCrew.run(inputs, obs)` is the single public entry point:

1. Asserts `crew_version` is set.
2. Loads agent specs (YAML), fetches each prompt as `agent.<name>` from Langfuse via `PromptLoader` (with YAML fallback). Only the LLM-text fields (`role`, `goal`, `backstory`, optional `system_template`/`prompt_template`/`response_template`) are pulled from the merged config.
3. Loads task specs (YAML), fetches each prompt as `task.<name>` from Langfuse the same way. Only `description` and `expected_output` are pulled. All other YAML keys are forwarded verbatim to `Task(**kwargs)` as wiring.
4. Builds a `crewai.Crew` with step/task callbacks injected by `get_crew_kwargs(obs)`.
5. Computes `prompt_meta`: per-agent and per-task prompt name/version/source plus deterministic `agents_signature` and `tasks_signature` strings.
6. Opens the root span (`obs.span(...)`), kicks off the crew, sets output, returns
   `{result, stdout, stderr, prompt_versions}`.

#### When to bump `crew_version`

Bump for any change to the *recipe*:

- `_agent_yaml_names` changes (add/remove/swap an agent)
- `_task_yaml_names` changes (reorder, add, or remove a task)
- `_format_result` semantics change
- The wired tool set changes
- An agent's or task's `prompt_key` is renamed (so a different Langfuse prompt resolves)
- Task-YAML wiring changes (agent assignment, `context`, `tools`, `output_pydantic`, ...)

Do **not** bump it for a Langfuse-side edit to an existing agent's or task's
prompt — those are already captured per-run in `agents_signature` and
`tasks_signature`.

### 4.2 Agent (configurable behavior)

An agent is two things layered:

- **YAML in `agents/<name>.yaml`** — wiring: `agent_name` (the bare key the
  Langfuse prompt resolves under, namespaced to `agent.<name>` at lookup
  time), `verbose`, `allow_delegation`, and a `fallback` dict with `role`,
  `goal`, `backstory`. Wiring keys never reach Langfuse.
- **Langfuse prompt** named `agent.<name>` (production label) — runtime
  source of truth for the LLM-text fields. The loader pulls only an
  allow-list (`role`, `goal`, `backstory`, optional `system_template` /
  `prompt_template` / `response_template`). Anything else Langfuse returns
  is dropped — a Langfuse edit cannot inject a different `llm`, `tools`, or
  `allow_delegation`.

This separation lets the team edit prompt text in Langfuse without code
deploys, while keeping wiring and a deterministic fallback in version
control.

### 4.3 Task

A task YAML carries **wiring at the top level** and a **`fallback:` block**
for the LLM-text fields:

```yaml
# tasks/research_task.yaml
task_name: research
agent: researcher
prompt_key: research_task          # bare key; resolves as "task.research_task" in Langfuse

# Any other CrewAI Task kwarg can be added here; loader forwards it to Task(**kwargs):
# async_execution: false
# max_retries: 2
# tools: [calculator]
# context: [other_task_name]       # resolver hook only (not implemented yet)
# output_pydantic: schemas:Result  # resolver hook only (not implemented yet)

fallback:
  description: 'Research the question: "{question}"'
  expected_output: A clear, concise answer with key points and 1-3 examples if applicable.
```

**Reserved keys** consumed by the loader: `task_name`, `agent`, `prompt_key`,
`fallback`. Everything else is generic wiring — forwarded verbatim to
`Task(**kwargs)`. Adding a new CrewAI Task feature is usually a zero-code
YAML edit.

**LLM-text fields** (`description`, `expected_output`) resolve as
`task.<prompt_key>` from Langfuse with the `fallback` block as the safety
net. As with agents, only these fields are pulled from the merged config —
wiring cannot be overridden via a Langfuse edit.

Input substitution (`{var}` interpolation from `inputs`) is applied to both
fields, using double-brace escaping so literal `{` / `}` in input values is
safe.

### 4.3.1 Concept-prefix namespacing

Langfuse prompt names follow `<concept>.<key>`, applied by the per-concept
loader (`_load_agents`, `_load_tasks`) before the `PromptLoader.get()` call:

| Concept | YAML field   | Bare key                | Langfuse name              |
|---------|--------------|-------------------------|----------------------------|
| Agent   | `agent_name` | `researcher`            | `agent.researcher`         |
| Task    | `prompt_key` | `research_task`         | `task.research_task`       |

A bare key containing `.` raises with a clear error pointing at the YAML
file — prevents accidental double-namespacing. Future concepts (`crew.`,
`tool.`, ...) get the same treatment.

### 4.4 Flow

A Flow is a `crewai.flow.flow.Flow[State]` subclass that holds typed state
(`pydantic.BaseModel`) and exposes one or more `@start()` methods. Each Flow
exists to:

1. Construct an `EnrichedConnectorManager` (observability with `RunContext`).
2. Instantiate its Crew, passing the connectors-factory.
3. Run the crew, flush spans, and populate the state model.

Each Flow subclass declares its own identity, propagated to every span:

```python
class ResearchFlow(Flow[ResearchState]):
    flow_name: ClassVar[str] = "researcher"
    flow_version: ClassVar[str] = "1.0.0"
```

`flow_name` / `flow_version` are independent from `crew_name` / `crew_version`:
the Crew's recipe lives on the crew class, while the Flow's recipe (topology,
state model, post-processing, which crew(s) it orchestrates) lives here.
Bump `flow_version` when the Flow body changes — even if the Crew didn't —
so traces can be bucketed by either axis.

Flows are what `crew_app.py` calls. They give the UI a uniform interface
(`flow.kickoff(inputs={...})`) regardless of crew internals.

### 4.5 PromptLoader

`core/prompts/loader.py` fetches a prompt by name + label from Langfuse and
merges its `config` dict over a fallback dict supplied by the caller. The
loader is **concept-agnostic** — the `agent.` / `task.` prefix is applied
by the caller (`crews/base.py`), not the loader. Future concepts plug in
without changing this file.

| Scenario | Behavior |
|---|---|
| Langfuse credentials missing | Use fallback silently (expected in local dev without `.env`). |
| Auth failure (401/403) | Log ERROR, use fallback. |
| Prompt not found (404) | Log WARNING, use fallback. |
| Langfuse unreachable | Log WARNING, use fallback. |
| Success | Returns `PromptResult(config, version=<langfuse>, name, label)`. |

Returned `version` is `"fallback"` when the registry is not used; otherwise
it is the stringified Langfuse version number. The merge is permissive at
the loader layer; callers are responsible for pulling only the fields they
trust to be Langfuse-managed (see `_pull_llm_text` in `crews/base.py`).

### 4.6 Observability

#### Connectors

`BaseConnector` (in `core/observability/base.py`) defines the contract every
observability backend implements:

```python
class BaseConnector(ABC):
    handles_step_callbacks: bool = True
    @property
    @abstractmethod
    def enabled(self) -> bool: ...
    @abstractmethod
    @contextmanager
    def span(self, name, span_type, input_data=None, metadata=None) -> Iterator[SpanHandle]: ...
    def flush(self) -> None: pass
    def update_run_context(self, context) -> None: pass
```

Concrete connectors:

- `LangfuseConnector` — opens Langfuse spans, propagates `RunContext` to trace metadata.
- `DatadogConnector` — gated by `DD_LLMOBS_ENABLED`; uses native CrewAI instrumentation
  (so `handles_step_callbacks=False`).

`ConnectorManager` fans `span()`/`flush()`/`update_run_context()` out across all
enabled connectors. `EnrichedConnectorManager` wraps it and injects `RunContext`
into every span's metadata + propagates context to each connector.

#### SpanHandle protocol

Connector `span()` yields a `SpanHandle`. Callers may call `set_output(...)` and
`mark_error()` **at most once each** before the context manager exits. The base
class does not merge multiple calls.

#### Run context

`RunContext` (in `core/observability/context/run_context.py`) is the run's
identity card. It is built once per crew run by `make_run_context(crew_name,
crew_version)` and propagated via `EnrichedConnectorManager` to every span.

| Field | Source | Notes |
|---|---|---|
| `session_id` | Streamlit tab (cached in `st.session_state`) | One per browser tab. |
| `run_id` | `uuid4()` per crew run | Also default for `workflow_id`. |
| `user_id` | `USER_ID` env var, falls back to `session_id` | |
| `environment` | `ENVIRONMENT` env var, default `dev` | Drives `config/environments/<env>.yaml` lookup. |
| `app_version` | `VERSION` file (fallback: `APP_VERSION` env, then `0.0.0`) | |
| `crew_name` | Caller-provided | E.g. `"researcher"`, `"fitness_training"`. |
| `flow_name` | Caller-provided (from `Flow.flow_name`) | Same as `crew_name` today (1:1 mapping); diverges when a flow orchestrates more than one crew. |
| `crew_version` | Pulled from the crew class | Crew recipe semver. |
| `flow_version` | Pulled from the flow class | Flow recipe semver. Independent of `crew_version`. |
| `deployment_sha` | `config/environments/<env>.yaml` or `DEPLOYMENT_SHA` env | |
| `model_version` | `config/environments/<env>.yaml` `model_defaults.default` or `MODEL_VERSION` env | |
| `workflow_id` | Settable, defaults to `run_id` | For grouping retries of the same logical workflow. |

### 4.7 Versioning model (three layers)

| Layer | Field(s) | Owner | When it changes |
|---|---|---|---|
| **Deployment** | `app_version` (`VERSION` file), `deployment_sha` | Build/release | Every deploy. |
| **Flow recipe** | `flow_version` (semver on the Flow class) | Flow authors | Manual PR bump when the Flow body changes: `@start`/`@listen`/`@router` topology, state-model fields, which crew(s) it orchestrates, or post-processing. |
| **Crew recipe** | `crew_version` (semver on the crew class) | Crew authors | Manual PR bump when the recipe (agent list, task order, tools, task-YAML wiring, formatting code) changes. |
| **Per-run prompt resolution** | `agents_signature` + `tasks_signature` (root-span metadata), plus per-prompt `prompt_name`/`prompt_version`/`prompt_source` entries | Runtime + Langfuse | Auto — whenever Langfuse serves a different version for any agent or task prompt. |

These four are independent filter axes in Langfuse: bucket by deployment,
flow recipe, crew recipe, or resolved prompts, in any combination. Flow and
crew versions move independently — changing a Flow's topology bumps
`flow_version` only; changing a Crew's agent list bumps `crew_version` only.

---

## 5. Runtime Sequence (one crew run)

```
crew_app.py
  └─ _run_flow(ResearchFlow, {"question": ...})
       └─ ResearchFlow.kickoff()
            └─ ResearchFlow.run_research()
                 ├─ make_run_context("researcher", crew_version=ResearchCrew.crew_version)
                 ├─ EnrichedConnectorManager(_get_connectors(), ctx)
                 └─ ResearchCrew().run(inputs, obs)
                      ├─ assert self.crew_version
                      ├─ PromptLoader.get("agent.researcher", fallback=...)        ← Langfuse
                      ├─ PromptLoader.get("task.research_task", fallback=...)      ← Langfuse
                      ├─ build Task(description.format(**safe_inputs), **wiring)
                      ├─ Crew(agents, tasks, **get_crew_kwargs(obs))
                      ├─ obs.span(crew_name, "chain", metadata={...prompt_meta, agents_signature, tasks_signature, ...})
                      │   └─ kickoff_crew(crew, obs)
                      │        ├─ captures stdout/stderr
                      │        ├─ runs CrewAI internals (each step/task creates child spans
                      │        │   via callbacks for Langfuse; native ddtrace patching for Datadog)
                      │        └─ returns (result, stdout, stderr)
                      ├─ _format_result(...)
                      ├─ root.set_output({"result": output})
                      └─ return {result, stdout, stderr, prompt_versions}
            └─ ResearchFlow state populated
       └─ obs.flush()        ← drains pending Langfuse / Datadog batches
```

---

## 6. Configuration

### Environment variables

#### Required

| Var | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY` | Langfuse SDK auth. |
| `LANGFUSE_SECRET_KEY` | Langfuse SDK auth. |
| `OPENAI_API_KEY` | Used by CrewAI agents (via OpenAI SDK). |

#### Optional — observability / runtime

| Var | Default | Purpose |
|---|---|---|
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Self-hosted Langfuse override. |
| `DD_LLMOBS_ENABLED` | unset (disabled) | `1`/`true` to enable Datadog LLMObs. |
| `DD_API_KEY` | — | Required when `DD_LLMOBS_ENABLED`. |
| `DD_SITE` | `datadoghq.com` | Datadog regional site. |
| `DD_LLMOBS_ML_APP` | `crew-streamlit` | Datadog "ML App" identifier. |
| `DD_LLMOBS_AGENTLESS_ENABLED` | `true` | Bypass local Datadog Agent. |
| `DD_LLMOBS_INTEGRATIONS_ENABLED` | `true` | Auto-instrument OpenAI etc. |
| `DD_TRACE_LLMOBS_IN_CODE` | `1` | Skip in-code init (rely on `ddtrace-run`). |
| `DD_ENV`, `DD_SERVICE` | — | Datadog tags. |
| `ENVIRONMENT` | `dev` | Selects `config/environments/<env>.yaml`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `USER_ID` | session id | Override per-run user identity. |
| `APP_VERSION` | `VERSION` file or `0.0.0` | Override app semver. |
| `MODEL_VERSION` | env config | Override the recorded model id. |
| `DEPLOYMENT_SHA` | env config | Override the recorded build SHA. |

`crew_app.py` calls `dotenv.load_dotenv()` at import, so a `.env` file in the
repo root is picked up automatically. **`.env` is gitignored — never commit it.**

### Environment-specific configuration

`config/environments/<env>.yaml` holds non-secret per-environment defaults:

```yaml
# config/environments/dev.yaml
environment: dev
deployment_sha: "local"
model_defaults:
  default: gpt-4o-mini
```

The file is selected by `ENVIRONMENT` (default `dev`). Cached after first read.

---

## 7. Extension Points

### Add a new crew

1. **Agent YAML** — create `agents/<agent>.yaml` with `agent_name`, `fallback`
   dict (role/goal/backstory).
2. **Task YAML(s)** — create `tasks/<task>.yaml` with `task_name`, `agent`,
   `prompt_key`, and a `fallback` block (`description`, `expected_output`).
   Any other CrewAI Task kwarg can be added at the top level and is forwarded
   to `Task(**kwargs)`.
3. **Crew class** — `crews/<name>_crew.py`:
   ```python
   class MyCrew(BaseCrew):
       crew_version = "1.0.0"
       @property
       def crew_name(self): return "crewai.my_crew"
       @property
       def _agent_yaml_names(self): return ["my_agent.yaml"]
       @property
       def _task_yaml_names(self): return ["my_task.yaml"]
   ```
4. **Flow** — `flows/<name>_flow.py` mirroring `ResearchFlow`.
5. **Langfuse prompts** — create one prompt per agent (`agent.<agent_name>`)
   and one per task (`task.<prompt_key>`), labeled `production`. Easiest:
   run `python scripts/seed_prompts.py` which walks both YAML directories.
   Without these, the YAML fallback is used.
6. **UI tab** — add a tab in `crew_app.py` that calls `_run_flow(MyFlow, {...})`.

### Add a new observability connector

1. Subclass `BaseConnector` in `core/observability/<name>_connector.py`.
2. Implement `enabled`, `span()`, and (optionally) `flush()` /
   `update_run_context()`.
3. If the backend has its own CrewAI instrumentation, set
   `handles_step_callbacks = False`.
4. Wire an instance into `_get_connectors()` in `crew_app.py`.

### Edit a prompt without a deploy

Works the same for **agent** (`agent.<name>`) and **task** (`task.<name>`)
prompts: edit in the Langfuse UI, promote to `production`. Active runs pick
it up after the cache TTL (`cache_ttl=300s` by default). The recipe
(`crew_version`) does not need to change; the new version shows up in
`agents_signature` / `tasks_signature` and per-prompt `prompt_version`
metadata on the trace.

Only the LLM-text fields are honored from Langfuse: extra keys you might
add to a prompt's Config dict are dropped at the loader. To change wiring
(`agent`, `tools`, `context`, `output_pydantic`, ...) edit the YAML and bump
`crew_version`.

---

## 8. User Instructions

### 8.1 First-time setup

**Prerequisites**

- Python 3.12 (matching `runtime.txt`)
- Git, Node.js (only if you want the `skills` CLI for `.agents/` skill packs)
- A Langfuse project and API keys
- An OpenAI API key

**Install**

```bash
# Clone
git clone <repo-url> langfuse-chat
cd langfuse-chat

# Virtualenv (recommended)
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# Dependencies
pip install -r requirements.txt
```

**Configure**

Create `.env` in the repo root:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
OPENAI_API_KEY=sk-...
ENVIRONMENT=dev

# Datadog LLMObs (optional)
# DD_LLMOBS_ENABLED=1
# DD_API_KEY=...
# DD_SITE=datadoghq.com
# DD_LLMOBS_ML_APP=crew-streamlit
```

`.env` is gitignored — keep it that way.

**Seed Langfuse prompts (optional but recommended)**

```bash
python scripts/seed_prompts.py
```

Walks both `agents/*.yaml` and `tasks/*.yaml`, creates one Langfuse prompt
per concept under its namespaced name (`agent.researcher`, `task.research_task`,
…) at version 1, labeled `production`. Without it, the app silently falls
back to the `fallback:` block of each YAML.

If you already have un-prefixed prompts in Langfuse (e.g. `researcher`), the
runtime will now fetch `agent.researcher` and miss them — either rename them
in the Langfuse UI to preserve history, or re-seed for a fresh start.

### 8.2 Run the app

```bash
py -3.12 -m streamlit run crew_app.py
```

The app opens at `http://localhost:8501` with three tabs:

- **Research** — single-agent researcher. Enter a question, click **Run research**.
- **Fitness Training** — three-agent pipeline (analyst → workout designer → nutrition
  advisor). Fill the form, click **Generate fitness plan**.
- **Experiments** — runs the researcher crew against a Langfuse dataset and
  logs each item as a Langfuse experiment.

Each result shows:

- The model output.
- A **Prompt sources** panel split into **Agents** and **Tasks** sections —
  each row shows the local name, the Langfuse prompt name (e.g.
  `agent.researcher`, `task.research_task`), its version, and whether it
  resolved from Langfuse (green) or the YAML fallback (yellow).
- An expandable **stdout / stderr** capture from the CrewAI run.

### 8.3 Run an experiment

In the **Experiments** tab:

1. Enter the dataset name (must already exist in your Langfuse project).
2. Enter an experiment-name prefix (a timestamp suffix is appended automatically).
3. Click **Run experiment**.

Results land in Langfuse → **Datasets** → `<dataset>` → **Experiments**.

You can also run experiments headless:

```bash
python scripts/run_experiment.py
```

### 8.4 Read traces in Langfuse

Each crew run produces one trace. Useful metadata fields to filter on:

| Field | Where it lives | What it tells you |
|---|---|---|
| `crew_name` | Trace metadata + tag | Which crew (`researcher`, `fitness_training`). |
| `flow_name` | Trace metadata + tag | Which Flow orchestrated this run (`researcher`, `fitness_training`). Today same as `crew_name`; diverges for multi-crew flows. |
| `crew_version` | Trace metadata | Which **crew recipe** version. Bumped manually. |
| `flow_version` | Trace metadata | Which **flow recipe** version. Bumped manually. Independent of `crew_version`. |
| `agents_signature` | Root span metadata | Which **agent prompt versions** resolved (e.g. `"researcher@6"`). |
| `tasks_signature` | Root span metadata | Which **task prompt versions** resolved (e.g. `"research@2"`). |
| `agent.<name>.prompt_name` / `prompt_version` / `prompt_source` | Root span metadata | Per-agent drill-down. `prompt_name` is the namespaced Langfuse name (e.g. `agent.researcher`). |
| `task.<name>.prompt_name` / `prompt_version` / `prompt_source` | Root span metadata | Per-task drill-down. |
| `app_version` | Trace metadata + tag | Which app build. |
| `deployment_sha` | Trace metadata + tag | Which commit. |
| `session_id` / `user_id` | Trace metadata | Group by browser tab / user. |

Recommended bucketing:

- Filter by `crew_version` → group runs of the same recipe.
- Filter by `agents_signature` + `tasks_signature` → group runs that
  resolved to the same prompt set (agents and tasks both stable).
- Combine all three → isolate "this recipe with these prompts" for clean
  A/B comparisons across deploys.

### 8.5 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| App fails at start with `Missing required environment variable` | `.env` not loaded or var missing. | Check `.env` in the repo root; verify the listed required vars are set. |
| **Prompt sources** panel shows yellow "YAML fallback" warnings | Langfuse prompt not found at `production` label, or Langfuse unreachable. | Promote the prompt in Langfuse UI, or run `python scripts/seed_prompts.py`. Logs at WARNING level identify which. |
| Yellow warnings after upgrading to namespaced prompts (`agent.X` / `task.X`) | Existing Langfuse prompts still under their old un-prefixed names. | Rename them in the Langfuse UI (preserves history) or re-seed via `scripts/seed_prompts.py`. |
| `task YAML ... missing required key(s) 'agent' or 'fallback'` | Task YAML still in pre-split shape (description at top level). | Move `description` / `expected_output` into a `fallback:` block; add `prompt_key`. |
| `prompt key '...' must be bare (no dots)` | A YAML `agent_name` or `prompt_key` contains a `.`. | Use bare names; the `agent.` / `task.` prefix is added automatically. |
| `assert self.crew_version` fails | Subclass missed setting `crew_version`. | Add `crew_version = "1.0.0"` to the crew class. |
| Datadog traces missing despite `DD_LLMOBS_ENABLED=1` | `ddtrace` not installed, or `DD_API_KEY` missing. | `pip install -r requirements.txt`; verify Datadog env vars. The boolean `_DD_LLMOBS_ACTIVE` in `crew_app.py` reflects whether init succeeded. |
| Streamlit cache-miss errors inside CrewAI's ThreadPoolExecutor | Cached resources first resolved in a background thread. | `_run_flow` already pre-resolves `_get_langfuse()` and `_get_connectors()` in the main thread; do not move that resolution. |
| `npx skills add ...` reinstalls flatly under `.agents/skills/<name>/` | The `skills` CLI does not know about the per-vendor subfolder layout. | Manually move into `.agents/skills/<vendor>/` and re-point the Junction at `.claude/skills/<name>`. See `.agents/README.md`. |

### 8.6 Day-to-day workflow

- **Edit an agent or task prompt:** Langfuse UI → save the `agent.<name>`
  or `task.<name>` prompt → promote to `production`. No deploy needed.
- **Add an agent to a crew:** edit YAML + bump that crew's `crew_version`.
- **Change a task instruction:** edit it in Langfuse (preferred — no deploy).
  Or edit the `fallback:` block of `tasks/<name>.yaml` if you want the
  default-in-version-control to change too (no `crew_version` bump unless
  wiring changes).
- **Change task wiring** (agent assignment, tools, context, output schema,
  async, retries, ...): edit `tasks/<name>.yaml` at the top level (outside
  the `fallback:` block) and bump `crew_version`.
- **Change a Flow's topology or state model** (`@start`/`@listen`/`@router`
  edits, state fields, swapping which crew it kicks off, post-processing):
  bump the Flow class's `flow_version`. Do **not** bump `crew_version` —
  that's a separate axis.
- **Add a new crew:** see [§7 Extension Points](#7-extension-points).
- **Update skill packs (for AI coding agents):** `npx skills update`. Re-group
  under `.agents/skills/<vendor>/` if new skills landed flatly.

---

## 9. Related Docs

- `.agents/README.md` — Skill-pack layout & how teammates restore via
  `npx skills experimental_install`.
- `core/prompts/loader.py` (module docstring) — Langfuse prompt setup details.
- `crew_app.py` (module docstring) — quick-reference of required env + run command.
- `crews/base.py` (`BaseCrew.crew_version` comment) — rules for when to bump.
