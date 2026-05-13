# Revio Multi-Crew Runner — System Documentation

## Overview

Revio Multi-Crew Runner is a Streamlit application that runs multi-agent AI workflows (crews) built with CrewAI. Every run is fully observable: spans are sent to **Langfuse** (and optionally **Datadog LLMObs**). Agent prompts are managed at runtime via **Langfuse Prompt Management**, so prompt versions can be promoted without redeploying.

**Key capabilities**

| Capability | Detail |
|---|---|
| Multi-crew UI | Streamlit tabs for Research and Fitness Training |
| Evaluation | Built-in Experiments tab; CLI runner for batch eval |
| Observability | Langfuse traces + Datadog LLMObs; both optional/pluggable |
| Prompt management | Prompts live in Langfuse; YAML fallback if Langfuse is unreachable |
| Prompt versioning | Each trace records `prompt_name`, `prompt_version`, `prompt_label` |
| Sessions | One Langfuse session per crew run; browser tab ID as user_id |

---

## Repository Layout

```
crew_app.py              # Streamlit entry point
run_experiment.py        # CLI experiment runner (batch eval)
seed_prompts.py          # One-time Langfuse prompt bootstrap
requirements.txt         # Pinned dependency versions

core/
  crews/
    __init__.py          # CREWS registry {name: class}
    base.py              # BaseCrew abstract class
    common.py            # kickoff_crew(), extract_question()
    researcher.py        # ResearcherCrew
    researcher.yaml      # Researcher agent/task config
    fitness_training.py  # FitnessTrainingCrew
    fitness_training.yaml# Fitness agents/tasks/output_sections config

  observability/
    __init__.py          # ConnectorManager (fan-out to all connectors)
    base.py              # BaseConnector, SpanHandle, ObsManager protocol
    langfuse_connector.py# Langfuse implementation
    datadog_connector.py # Datadog LLMObs implementation
    context/
      __init__.py        # Re-exports EnrichedConnectorManager, make_run_context
      enriched.py        # EnrichedConnectorManager (injects RunContext into spans)
      callbacks.py       # CrewAI step_callback / task_callback wrappers
      run_context.py     # RunContext dataclass
      session.py         # make_run_context() factory

  prompts/
    __init__.py          # Re-exports PromptLoader, PromptResult
    loader.py            # PromptLoader (Langfuse fetch + YAML fallback)
```

---

## Architecture

```
crew_app.py (Streamlit)
    │
    ├─ _get_langfuse()              @st.cache_resource — one client per process
    ├─ _get_connectors()            ConnectorManager([LangfuseConnector, DatadogConnector])
    │
    └─ _run_crew(crew_name, inputs)
           │
           ├─ make_run_context(crew_name)    → RunContext (session_id, run_id, user_id, ...)
           ├─ EnrichedConnectorManager(connectors, context)
           │       injects RunContext into every span's metadata
           │       pushes session_id/user_id to each connector via update_run_context()
           │       exposes crew_callbacks for CrewAI step/task hooks
           │
           └─ CREWS[crew_name]().run(inputs, obs)
                   │
                   ├─ PromptLoader.get(name, fallback=yaml_spec)
                   │       fetches prompt from Langfuse (label="production")
                   │       merges {**yaml_fallback, **langfuse_config}
                   │       returns PromptResult(config, version, name, label)
                   │
                   ├─ crewai.Agent(**prompt.config)   ← role/goal/backstory from Langfuse
                   ├─ crewai.Crew(agents, tasks, step_callback, task_callback)
                   │
                   └─ obs.span("crewai.research", "chain", metadata={...prompt versions...})
                           └─ kickoff_crew(crew, obs)
                                   └─ obs.span("crew.kickoff", "span")
                                           step_callback → obs.span("agent.step", "agent")
                                           task_callback → obs.span("task.complete", "span")
```

---

## Crews

### ResearcherCrew (`core/crews/researcher.py`)

Single-agent general-purpose research crew.

**Crew name**: `crewai.researcher`

**Config file**: `core/crews/researcher.yaml`

**Agents**

| Name | Role | Langfuse prompt key |
|---|---|---|
| `agent` | Researcher | `researcher_agent` |

**Inputs**

| Field | Type | Description |
|---|---|---|
| `question` | str | The research question to answer |

**Output**

| Field | Description |
|---|---|
| `result` | Plain-text research answer |
| `stdout` | Captured stdout from crew execution |
| `stderr` | Captured stderr from crew execution |

---

### FitnessTrainingCrew (`core/crews/fitness_training.py`)

Three-agent crew that produces a full personalized fitness plan (analysis → workout → nutrition).

**Crew name**: `crewai.fitness_training`

**Config file**: `core/crews/fitness_training.yaml`

**Agents**

| Name | Role | Langfuse prompt key |
|---|---|---|
| `fitness_analyst` | Fitness Analyst | `fitness_fitness_analyst` |
| `workout_designer` | Workout Program Designer | `fitness_workout_designer` |
| `nutrition_advisor` | Nutrition Advisor | `fitness_nutrition_advisor` |

**Inputs**

| Field | Type | Description |
|---|---|---|
| `goals` | str | User's fitness goals |
| `fitness_level` | str | `beginner` / `intermediate` / `advanced` |
| `equipment` | str | Available equipment |
| `time_per_week` | int | Hours per week available for training |
| `limitations` | str | Injuries or restrictions (optional) |

**Output**

| Field | Description |
|---|---|
| `result` | Markdown fitness plan (3 sections stitched from task outputs) |
| `stdout` | Captured stdout |
| `stderr` | Captured stderr |

---

### Adding a New Crew

1. Create `core/crews/<name>.py` — subclass `BaseCrew`, implement `crew_name` and `run()`.
2. Create `core/crews/<name>.yaml` — define `agents` and `tasks`.
3. Register in `core/crews/__init__.py` → `CREWS` dict.
4. Add seed entries to `seed_prompts.py` → `PROMPTS` list.
5. Add a UI tab in `crew_app.py`.

---

## Observability

### Span Hierarchy (per run)

```
crewai.research          [chain]   root span; input + output + prompt_meta in metadata
  crew.kickoff           [span]    wraps crew.kickoff(); captures stdout/stderr
    agent.step           [agent]   fires for every CrewAI reasoning step (truncated to 2000 chars)
    task.complete        [span]    fires when each Task finishes
```

### ConnectorManager (`core/observability/__init__.py`)

Fan-out coordinator. Receives a list of `BaseConnector` instances; on every `span()` call it opens a span on all enabled connectors simultaneously. Returns a `MultiSpanHandle` that broadcasts `update()` to all.

### EnrichedConnectorManager (`core/observability/context/enriched.py`)

Thin wrapper over `ConnectorManager`. On construction it calls `base.update_run_context(context)` on each connector and merges `RunContext.as_metadata()` into every span's metadata automatically.

Also exposes `crew_callbacks: CrewCallbacks` — the `get_crew_kwargs(obs)` helper in `core/crews/common.py` uses duck-typing (`hasattr(obs, "crew_callbacks")`) to wire callbacks if the addon is active.

### LangfuseConnector (`core/observability/langfuse_connector.py`)

Uses Langfuse SDK v4's `start_as_current_observation()`. For the first span per run it also calls `propagate_attributes(session_id=..., user_id=..., tags=[...])` inside the span context, propagating values to all child spans via OpenTelemetry baggage.

| Span field | Value |
|---|---|
| `name` | Span name (e.g. `crewai.research`) |
| `as_type` | Langfuse type (`chain`, `span`, `agent`) |
| `input` | `input_data` dict |
| `metadata` | Merged RunContext + crew-specific metadata including prompt versions |
| `output` | Set via `span_handle.update(output=...)` |

### DatadogConnector (`core/observability/datadog_connector.py`)

Wraps Datadog `ddtrace.llmobs.LLMObs`. Maps internal span types to Datadog span kinds:

| Internal type | Datadog kind |
|---|---|
| `chain` | `workflow` |
| `span` | `task` |
| `agent` | `agent` |
| `tool` | `tool` |
| `generation` | `llm` |

Active only when `DD_LLMOBS_ENABLED=1`. Uses `agentless_enabled=True` for Streamlit Cloud (no Datadog Agent sidecar).

### RunContext (`core/observability/context/run_context.py`)

Carries per-run identity metadata propagated to every span.

| Field | Default | Source |
|---|---|---|
| `session_id` | fresh UUID per run | `make_run_context()` |
| `run_id` | fresh UUID per run | `make_run_context()` |
| `user_id` | `USER_ID` env or Streamlit session UUID | `make_run_context()` |
| `environment` | `ENVIRONMENT` env (default `dev`) | env var |
| `app_version` | `APP_VERSION` env (default `1.0.0`) | env var |
| `crew_name` | passed from `_run_crew()` | caller |

`as_tags()` → Langfuse tag list: `["env:dev", "crew:researcher", "version:1.0.0"]`

`as_dd_tags()` → Datadog tag dict: `{"env": "dev", "crew": "researcher", "version": "1.0.0"}`

---

## Prompt Management

### Architecture

```
GitHub YAML  ──►  YAML fallback  ──►  deterministic default contract
Langfuse UI  ──►  runtime config ──►  rapidly-evolving AI behavior
```

YAML defines the starting-point defaults. Langfuse overrides them at runtime. Crews always work — even if Langfuse is unreachable — because the YAML fallback is always applied first.

### PromptLoader (`core/prompts/loader.py`)

```python
loader = PromptLoader()
prompt = loader.get(
    "researcher_agent",         # Langfuse prompt name
    fallback=yaml_agent_spec,   # dict from YAML: role, goal, backstory
    label="production",         # Langfuse rollout label
    cache_ttl=300,              # seconds (5-min local cache in Langfuse SDK)
)
agent = Agent(**prompt.config)
```

**Resolution logic**

1. If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set, fetch from Langfuse.
2. Merge: `{**yaml_fallback, **langfuse_config}` — Langfuse values override YAML.
3. On any failure (network, prompt not found, missing env vars) → use YAML fallback silently.

**PromptResult fields**

| Field | Type | Description |
|---|---|---|
| `config` | dict | Merged agent config (`role`, `goal`, `backstory`) |
| `version` | str | Langfuse version number, or `"fallback"` |
| `name` | str | Prompt name |
| `label` | str | Langfuse label used |

### Langfuse Prompts (name → crew)

| Prompt name | Crew | Agent |
|---|---|---|
| `researcher_agent` | ResearcherCrew | `agent` |
| `fitness_fitness_analyst` | FitnessTrainingCrew | `fitness_analyst` |
| `fitness_workout_designer` | FitnessTrainingCrew | `workout_designer` |
| `fitness_nutrition_advisor` | FitnessTrainingCrew | `nutrition_advisor` |

### Promoting a New Prompt Version

1. Open Langfuse UI → **Prompts** → select prompt.
2. Click **+ New version** → edit `role`, `goal`, `backstory` in the Config block.
3. Move the `production` label to the new version.
4. Within ~5 minutes (cache TTL), all running crew instances pick up the new version automatically.
5. Every Langfuse trace records `agent.<name>.prompt_version` in metadata — compare versions across runs in the Langfuse Traces view.

### Champion / Challenger Testing

- Promote the challenger to `staging` label.
- In `PromptLoader.get()` pass `label="staging"` for the experiment run (or run `run_experiment.py`).
- Compare traces in Langfuse Experiments view.
- When satisfied, move the `production` label.

### Seeding Initial Prompts (`seed_prompts.py`)

Run once after first deploy or when adding a new crew:

```
py -3.12 seed_prompts.py
```

Creates version 1 of all 4 prompts in Langfuse, labeled `production`. Idempotent — re-running creates new versions rather than overwriting.

---

## Experiments

### In-App (Experiments Tab)

- Enter a Langfuse dataset name (default: `crew-research-eval`) and experiment prefix.
- Runs ResearcherCrew against every dataset item with `max_concurrency=1`.
- Each item gets its own `EnrichedConnectorManager` + fresh `RunContext`.
- Results logged as Langfuse experiment runs; compare in Langfuse → Datasets → Experiments.
- The form is locked while running (no double-submit).

### CLI (`run_experiment.py`)

```
py -3.12 run_experiment.py
```

Same logic as the UI tab but runs from the terminal. Useful for CI pipelines.

---

## Environment Variables

### Required

| Variable | Used by |
|---|---|
| `LANGFUSE_PUBLIC_KEY` | LangfuseConnector, PromptLoader, experiments |
| `LANGFUSE_SECRET_KEY` | LangfuseConnector, PromptLoader, experiments |
| `OPENAI_API_KEY` | CrewAI (passed to OpenAI via crewai internals) |

### Optional — Langfuse

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Self-hosted Langfuse URL |

### Optional — Identity / Environment

| Variable | Default | Description |
|---|---|---|
| `USER_ID` | Streamlit session UUID | User ID propagated to Langfuse traces |
| `ENVIRONMENT` | `dev` | Tag on all spans (`env:dev`, `env:prod`, etc.) |
| `APP_VERSION` | `1.0.0` | Tag on all spans (`version:1.0.0`) |
| `SESSION_ID` | auto | Override session ID for non-Streamlit runs |

### Optional — Datadog LLMObs

| Variable | Default | Description |
|---|---|---|
| `DD_LLMOBS_ENABLED` | (off) | Set to `1` to activate Datadog connector |
| `DD_API_KEY` | — | Datadog API key |
| `DD_SITE` | `datadoghq.com` | Datadog site |
| `DD_LLMOBS_ML_APP` | `crew-streamlit` | ML app name in Datadog |
| `DD_LLMOBS_AGENTLESS_ENABLED` | `true` | Required on Streamlit Cloud (no agent sidecar) |
| `DD_ENV` | — | Datadog environment tag |
| `DD_SERVICE` | `crew-streamlit` | Datadog service name |
| `DD_TRACE_LLMOBS_IN_CODE` | `1` | Set to `0` if using `ddtrace-run` externally |

### Optional — Misc

| Variable | Default | Description |
|---|---|---|
| `CREWAI_TELEMETRY_OPT_OUT` | `true` (forced) | Disables CrewAI's built-in telemetry |
| `SEED_LABEL` | `production` | Label used by `seed_prompts.py` |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars (or create .env)
cp .env.example .env  # fill in LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, OPENAI_API_KEY

# Seed Langfuse prompts (first time only)
py -3.12 seed_prompts.py

# Start the app
py -3.12 -m streamlit run crew_app.py
```

---

## Deployment (Streamlit Cloud)

**Secrets** — add all required env vars in the Streamlit Cloud app settings under **Secrets** (TOML format).

**`requirements.txt` notes**

The Streamlit Cloud installer (`uv`) strictly enforces version bounds. The following pins prevent known conflicts:

| Package | Pinned version | Reason |
|---|---|---|
| `crewai` | `==1.14.4` | Exact version; newer versions may change APIs |
| `chromadb` | `==1.1.1` | crewai requires `~=1.1.0`; `>=1.2.0` conflicts |
| `pydantic` | `>=2.11.9,<2.13` | crewai needs `>=2.11.9`; chromadb transitively pulls 2.11.7 |
| `openai` | `>=2.30.0,<3.0.0` | crewai requires `>=2.30.0`; older cap (`<2.0`) caused failure |
| `langfuse` | `>=4.0.0,<5.0.0` | SDK v4 required; v3 no longer delivers traces to Langfuse Cloud |

**Hot-reload fix** — `crew_app.py` purges all `core.*` entries from `sys.modules` before importing them. This prevents `KeyError: 'core'` on Streamlit Cloud's hot-reload cycles.

**Datadog on Streamlit Cloud** — must use `agentless_enabled=True` and `DD_TRACE_ENABLED=0` since there is no Datadog Agent sidecar. LLMObs is initialized before any OTel-using imports to avoid `"Overriding of current TracerProvider is not allowed"` warnings.

---

## Adding a New Observability Connector

1. Create `core/observability/<name>_connector.py`.
2. Subclass `BaseConnector`; implement `enabled`, `span()`, `flush()`, `update_run_context()`.
3. Add an instance to the list in `_get_connectors()` in `crew_app.py`.

No changes to crews, PromptLoader, or RunContext needed.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `ObsManager` as a Protocol | Crews import nothing from `core.observability`; structural typing avoids circular imports on Streamlit hot-reload |
| `TYPE_CHECKING` guards in `enriched.py` | Prevents re-import of `core.observability` while `crew_app.py` is still loading it |
| One session per run (not per browser tab) | Each crew run is a distinct Langfuse session; user_id still correlates the tab across runs |
| `{**yaml_fallback, **langfuse_config}` merge | YAML provides a complete working default; Langfuse overrides selectively — any missing Langfuse key transparently uses YAML |
| `cache_ttl_seconds=300` on `get_prompt()` | Langfuse SDK caches locally for 5 min; new label promotion is live within one cache window |
| `@st.cache_resource` for Langfuse client | Single client per Streamlit worker process; avoids re-authenticating on every rerun |
| `max_concurrency=1` in experiments | Prevents parallel CrewAI runs from interleaving stdout/stderr and LLM rate-limit errors |
