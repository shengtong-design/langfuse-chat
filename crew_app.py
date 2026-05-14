"""
Streamlit app: multi-crew runner with flows, crews, and modular observability.

Architecture:
  flows/    — CrewAI Flow entry points (research_flow, fitness_flow)
  crews/    — Crew implementations (research_crew, fitness_crew)
  agents/   — Agent YAML config (metadata only; prompts live in Langfuse)
  tasks/    — Task YAML config (descriptions + expected outputs)
  core/observability/ — Langfuse + Datadog connectors

To add a new connector: create core/observability/<name>.py, subclass BaseConnector,
add an instance to _get_connectors() below.

To add a new crew: create agents/<n>.yaml, tasks/<n>_task.yaml,
crews/<n>_crew.py (subclass BaseCrew), add a flow in flows/<n>_flow.py.

Required env vars:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, OPENAI_API_KEY

Datadog (optional):
  DD_LLMOBS_ENABLED=1, DD_API_KEY, DD_SITE, DD_LLMOBS_ML_APP

Run:
  py -3.12 -m streamlit run crew_app.py
"""

from __future__ import annotations

import logging
import os

# CrewAI 1.14.x telemetry opt-out — checks CREWAI_DISABLE_TELEMETRY (and
# OTEL_SDK_DISABLED) at import time. The legacy CREWAI_TELEMETRY_OPT_OUT name
# from earlier versions is silently ignored, which is how we ended up with
# "telemetry.crewai.com: Connection refused" log noise on Streamlit Cloud.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _init_datadog_llmobs() -> bool:
    """Enable Datadog LLMObs with native ddtrace integrations turned on.

    Native integrations auto-instrument CrewAI / OpenAI / LiteLLM — this is
    the "default OpenTelemetry to Datadog" path; we no longer hand-roll
    spans via DatadogConnector. The DatadogConnector class still exists
    in ``core/observability/datadog_connector.py`` if it ever needs to
    be revived, but it is intentionally not wired into _get_connectors()
    so Datadog only sees ddtrace's native spans (no double instrumentation,
    no RunContext leak from our code path). Override with
    ``DD_LLMOBS_INTEGRATIONS_ENABLED=false`` to turn auto-patching off.

    Deliberately called *after* the top-level package imports so ddtrace's
    import wrapper doesn't sit between Python's loader and our own modules
    during cold start — we hit an intermittent
    ``KeyError: 'core.observability.base'`` on Streamlit Cloud when ddtrace
    was initialised first. By the time this runs, ``crewai`` is already
    in ``sys.modules`` so ddtrace's patcher operates on the live module
    object rather than wrapping our re-imports.
    """
    if os.getenv("DD_LLMOBS_ENABLED", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    if os.getenv("DD_TRACE_LLMOBS_IN_CODE", "1").strip().lower() in ("0", "false", "no"):
        return True
    try:
        os.environ.setdefault("DD_TRACE_ENABLED", "0")
        logging.getLogger("ddtrace").setLevel(logging.ERROR)
        from ddtrace.llmobs import LLMObs
        LLMObs.enable(
            ml_app=os.getenv("DD_LLMOBS_ML_APP", "crew-streamlit"),
            api_key=os.getenv("DD_API_KEY"),
            site=os.getenv("DD_SITE", "datadoghq.com"),
            agentless_enabled=os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on"),
            env=os.getenv("DD_ENV"),
            service=os.getenv("DD_SERVICE", "crew-streamlit"),
            integrations_enabled=os.getenv("DD_LLMOBS_INTEGRATIONS_ENABLED", "true").strip().lower() not in ("0", "false", "no"),
        )
        return True
    except ModuleNotFoundError:
        return False


import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from core.observability import ConnectorManager
from core.observability.langfuse_connector import LangfuseConnector
from crews.common import extract_question
from flows import FitnessFlow, ResearchFlow

# Initialise ddtrace AFTER package imports — see _init_datadog_llmobs docstring.
_DD_LLMOBS_ACTIVE = _init_datadog_llmobs()


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
    # Not cached: connectors are cheap; LangfuseConnector reuses _get_langfuse()
    # which IS cached. Fresh instances every render avoid stale class attrs
    # surviving hot-reloads. Datadog is intentionally absent — ddtrace's native
    # CrewAI/OpenAI integrations (enabled in _init_datadog_llmobs) produce all
    # the Datadog spans now; our DatadogConnector is dormant in
    # core/observability/datadog_connector.py if it needs to be revived.
    return ConnectorManager([
        LangfuseConnector(_get_langfuse()),
    ])


def _run_flow(flow_cls, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate a flow, kick it off, and return the result dict.

    Both cached resources are resolved here (main Streamlit thread) so that
    the cache-miss path never runs inside CrewAI's background ThreadPoolExecutor,
    which would fail in Streamlit because @st.cache_resource requires a script
    run context on the first call.
    """
    connectors = _get_connectors()
    flow = flow_cls(connectors_factory=lambda: connectors)
    result = flow.kickoff(inputs=inputs)
    if isinstance(result, dict):
        return result
    return {
        "result": getattr(result, "result", str(result)),
        "stdout": getattr(result, "stdout", ""),
        "stderr": getattr(result, "stderr", ""),
        "prompt_versions": getattr(result, "prompt_versions", {}),
    }


def _show_output(data: Dict[str, Any], heading: str = "Result", markdown: bool = False) -> None:
    """Render crew result and collapsible stdout/stderr."""
    st.subheader(heading)
    (st.markdown if markdown else st.write)(data.get("result", ""))
    pv = data.get("prompt_versions", {})

    def _render_section(prefix: str, label: str) -> None:
        suffix = ".prompt_source"
        names = sorted({
            k[len(prefix):-len(suffix)]
            for k in pv if k.startswith(prefix) and k.endswith(suffix)
        })
        if not names:
            return
        st.markdown(f"**{label}**")
        for n in names:
            prompt_name = pv.get(f"{prefix}{n}.prompt_name", n)
            version = pv.get(f"{prefix}{n}.prompt_version", "?")
            source = pv.get(f"{prefix}{n}.prompt_source", "yaml_fallback")
            if source == "langfuse":
                st.success(f"`{n}` — Langfuse **{prompt_name}** v{version}", icon="✅")
            else:
                st.warning(f"`{n}` — YAML fallback (prompt `{prompt_name}` not found in Langfuse)", icon="⚠️")

    if any(k.endswith(".prompt_source") for k in pv):
        with st.expander("Prompt sources", expanded=True):
            _render_section("agent.", "Agents")
            _render_section("task.", "Tasks")
    with st.expander("stdout / stderr", expanded=False):
        st.code(data.get("stdout") or "", language="text")
        if data.get("stderr"):
            st.code(data["stderr"], language="text")


# ── UI ────────────────────────────────────────────────────────────────────────

print("[crew_app] rendering UI", flush=True)
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
            with st.spinner("Running researcher crew..."):
                data = _run_flow(ResearchFlow, {"question": question.strip()})
            _show_output(data)
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
        health_report = st.file_uploader(
            "Health report (optional)",
            type=["txt", "md", "pdf"],
            help="Upload a recent health report; the fitness analyst will read it via the health_report_reader tool.",
        )
        fitness_btn = st.form_submit_button("Generate fitness plan", type="primary")

    if fitness_btn:
        if not goals.strip() or not equipment.strip():
            st.warning("Please fill in goals and available equipment.")
        else:
            health_report_path = ""
            if health_report is not None:
                suffix = Path(health_report.name).suffix.lower() or ".bin"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(health_report.getvalue())
                tmp.close()
                health_report_path = tmp.name
            try:
                with st.spinner("Generating your personalized fitness plan (3 agents)..."):
                    data = _run_flow(FitnessFlow, {
                        "goals": goals.strip(),
                        "fitness_level": fitness_level,
                        "equipment": equipment.strip(),
                        "time_per_week": time_per_week,
                        "limitations": limitations.strip() or "None specified",
                        "health_report_path": health_report_path,
                    })
                _show_output(data, heading="Your Personalized Fitness Plan", markdown=True)
            except Exception as e:
                st.error(f"Failed: {e}")
            finally:
                if health_report_path:
                    Path(health_report_path).unlink(missing_ok=True)

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

                connectors = _get_connectors()  # resolve in main thread before run_experiment threads start

                def _task(item):
                    q = extract_question(item.input)
                    flow = ResearchFlow(
                        connectors_factory=lambda: connectors,
                    )
                    result = flow.kickoff(inputs={"question": q})
                    return result.get("result", "") if isinstance(result, dict) else str(result)

                with st.spinner(f"Running {len(items)} items — this may take a while..."):
                    langfuse_client.run_experiment(
                        name=experiment_name,
                        run_name=experiment_name,
                        data=items,
                        task=_task,
                        max_concurrency=1,
                        metadata={"framework": "crewai", "runner": "crew_app.py"},
                    )
                st.success(f"Experiment **{experiment_name}** complete!")
                st.caption(f"Check Langfuse → Datasets → {dataset_name} → Experiments")
            except Exception as e:
                st.error(f"Failed: {e}")
            finally:
                st.session_state.experiment_running = False
