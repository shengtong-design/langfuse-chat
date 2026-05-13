"""
Streamlit app: multi-crew runner with modular observability.

Crews (core/crews/):
  - researcher        General Q&A research
  - fitness_training  Personalized fitness plan (analysis + workout + nutrition)

Observability connectors (core/observability/):
  - Langfuse   always active when LANGFUSE_PUBLIC_KEY is set
  - Datadog    active when DD_LLMOBS_ENABLED=1

To add a new connector: create core/observability/<name>.py, subclass BaseConnector,
add an instance to _get_connectors() below.

To add a new crew: create core/crews/<name>.py, subclass BaseCrew,
add it to core/crews/__init__.py CREWS dict.

Required env vars:
  - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
  - OPENAI_API_KEY

Datadog (optional):
  - DD_LLMOBS_ENABLED=1
  - DD_API_KEY, DD_SITE, DD_LLMOBS_ML_APP, DD_LLMOBS_AGENTLESS_ENABLED=1

Run:
  py -3.12 -m streamlit run crew_app.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


def _init_datadog_llmobs() -> bool:
    """Enable Datadog LLMObs before any OTel-using imports claim the TracerProvider."""
    if os.getenv("DD_LLMOBS_ENABLED", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    if os.getenv("DD_TRACE_LLMOBS_IN_CODE", "1").strip().lower() in ("0", "false", "no"):
        return True  # ddtrace-run handles init externally
    try:
        os.environ.setdefault("DD_TRACE_ENABLED", "0")
        from ddtrace.llmobs import LLMObs
        LLMObs.enable(
            ml_app=os.getenv("DD_LLMOBS_ML_APP", "crew-streamlit"),
            api_key=os.getenv("DD_API_KEY"),
            site=os.getenv("DD_SITE", "datadoghq.com"),
            agentless_enabled=os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on"),
            env=os.getenv("DD_ENV"),
            service=os.getenv("DD_SERVICE", "crew-streamlit"),
            integrations_enabled=os.getenv("DD_LLMOBS_INTEGRATIONS_ENABLED", "false").strip().lower() not in ("0", "false", "no"),
        )
        return True
    except ModuleNotFoundError:
        return False


_DD_LLMOBS_ACTIVE = _init_datadog_llmobs()

from datetime import datetime

import streamlit as st
from opentelemetry import trace

from core.crews import CREWS
from core.observability import ConnectorManager
from core.observability.context import EnrichedConnectorManager, make_run_context
from core.observability.datadog_connector import DatadogConnector
from core.observability.langfuse_connector import LangfuseConnector


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@st.cache_resource
def _get_langfuse():
    from langfuse import Langfuse
    return Langfuse(
        public_key=_require_env("LANGFUSE_PUBLIC_KEY"),
        secret_key=_require_env("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


def _get_connectors() -> ConnectorManager:
    return ConnectorManager([
        LangfuseConnector(_get_langfuse()),
        DatadogConnector(_DD_LLMOBS_ACTIVE),
    ])


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CrewAI Runner", layout="wide")
st.title("Revio Multi-Crew Runner")

tab_research, tab_fitness, tab_experiment = st.tabs(["Research", "Fitness Training", "Experiments"])

# ── Research ──────────────────────────────────────────────────────────────────
with tab_research:
    question = st.text_area(
        "Research question",
        placeholder='e.g. "What is AI?"',
        height=120,
        key="research_q",
    )
    research_btn = st.button(
        "Run research",
        type="primary",
        disabled=not question.strip(),
        key="research_btn",
    )

    if research_btn:
        try:
            obs = EnrichedConnectorManager(_get_connectors(), make_run_context("researcher"))
            with st.spinner("Running researcher crew..."):
                data = CREWS["researcher"]().run({"question": question.strip()}, obs)
            obs.flush()
            st.subheader("Result")
            st.write(data["result"])
            with st.expander("stdout / stderr", expanded=False):
                st.code(data.get("stdout") or "", language="text")
                if data.get("stderr"):
                    st.code(data["stderr"], language="text")
        except Exception as e:
            st.error(f"Failed: {e}")

# ── Fitness Training ──────────────────────────────────────────────────────────
with tab_fitness:
    with st.form("fitness_form"):
        goals = st.text_input(
            "Fitness goals",
            placeholder='e.g. "Build muscle and improve overall strength"',
        )
        fitness_level = st.selectbox(
            "Current fitness level",
            ["beginner", "intermediate", "advanced"],
        )
        equipment = st.text_input(
            "Available equipment",
            placeholder='e.g. "Full gym access" or "Bodyweight only"',
        )
        time_per_week = st.slider("Hours available per week", min_value=1, max_value=20, value=5)
        limitations = st.text_input(
            "Limitations / injuries (optional)",
            placeholder='e.g. "Minor lower back sensitivity"',
        )
        fitness_btn = st.form_submit_button("Generate fitness plan", type="primary")

    if fitness_btn:
        if not goals.strip() or not equipment.strip():
            st.warning("Please fill in goals and available equipment.")
        else:
            try:
                obs = EnrichedConnectorManager(_get_connectors(), make_run_context("fitness_training"))
                with st.spinner("Generating your personalized fitness plan (3 agents)..."):
                    data = CREWS["fitness_training"]().run(
                        {
                            "goals": goals.strip(),
                            "fitness_level": fitness_level,
                            "equipment": equipment.strip(),
                            "time_per_week": time_per_week,
                            "limitations": limitations.strip() or "None specified",
                        },
                        obs,
                    )
                obs.flush()
                st.subheader("Your Personalized Fitness Plan")
                st.markdown(data["result"])
                with st.expander("stdout / stderr", expanded=False):
                    st.code(data.get("stdout") or "", language="text")
                    if data.get("stderr"):
                        st.code(data["stderr"], language="text")
            except Exception as e:
                st.error(f"Failed: {e}")

# ── Experiments ───────────────────────────────────────────────────────────────
with tab_experiment:
    st.caption("Runs the researcher crew against a Langfuse dataset and logs results for evaluation.")

    if "experiment_running" not in st.session_state:
        st.session_state.experiment_running = False

    with st.form("experiment_form"):
        dataset_name = st.text_input("Dataset name", value="crew-research-eval")
        experiment_prefix = st.text_input("Experiment name prefix", value="crewai-researcher-v1")
        exp_btn = st.form_submit_button(
            "Experiment running..." if st.session_state.experiment_running else "Run experiment",
            type="primary",
            disabled=st.session_state.experiment_running,
        )

    if exp_btn and not st.session_state.experiment_running:
        if not dataset_name.strip():
            st.warning("Please enter a dataset name.")
        else:
            experiment_name = f"{experiment_prefix.strip()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            try:
                st.session_state.experiment_running = True
                langfuse_client = _get_langfuse()
                dataset = langfuse_client.get_dataset(dataset_name.strip())
                items = list(dataset.items)
                st.info(f"Found **{len(items)}** items in dataset `{dataset_name}`. Running as `{experiment_name}`...")

                crew = CREWS["researcher"]()

                def _task(item):
                    q = item.input
                    if isinstance(q, dict):
                        q = q.get("question") or q.get("query") or q.get("input") or str(q)
                    q = str(q)
                    trace.get_current_span().update_name(q)
                    # Fresh obs per item: same session_id (same browser tab) but new run_id.
                    item_obs = EnrichedConnectorManager(_get_connectors(), make_run_context("researcher"))
                    result = crew.run({"question": q}, item_obs)
                    item_obs.flush()
                    return result["result"]

                with st.spinner(f"Running {len(items)} items — this may take a while..."):
                    langfuse_client.run_experiment(
                        name=experiment_name,
                        run_name=experiment_name,
                        data=items,
                        task=_task,
                        max_concurrency=1,
                        metadata={"framework": "crewai", "runner": "crew_app.py"},
                    )

                _get_connectors().flush()
                st.success(f"Experiment **{experiment_name}** complete!")
                st.caption(f"Check Langfuse → Datasets → {dataset_name} → Experiments")
            except Exception as e:
                st.error(f"Failed: {e}")
            finally:
                st.session_state.experiment_running = False
