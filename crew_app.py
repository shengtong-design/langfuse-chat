"""
Streamlit app: run a CrewAI Researcher on a question and log to Langfuse (SDK v4).

Required env vars:
  - LANGFUSE_PUBLIC_KEY=...
  - LANGFUSE_SECRET_KEY=...
  - LANGFUSE_BASE_URL=https://cloud.langfuse.com  (or your self-hosted base url)

LLM provider for CrewAI (example):
  - OPENAI_API_KEY=...

Install deps:
  - py -3.12 -m pip install streamlit crewai langfuse ddtrace python-dotenv

Datadog LLM Observability (optional): set e.g.
  - DD_LLMOBS_ENABLED=1
  - DD_LLMOBS_ML_APP=crew-streamlit
  - DD_API_KEY=... (agentless)
  - DD_SITE=datadoghq.com (or your site)
  - DD_LLMOBS_AGENTLESS_ENABLED=1

Alternatively run with: ddtrace-run py -3.12 -m streamlit run crew_app.py
(do not combine ddtrace-run with LLMObs.enable() in code — use one or the other).

Run:
  - py -3.12 -m streamlit run crew_app.py
"""

from __future__ import annotations

# Keep CrewAI from initializing its own telemetry.
import os

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    # App can still run if env vars are set outside a .env file.
    pass


def _init_datadog_llmobs() -> bool:
    """Enable Datadog LLMObs as early as possible, before any OTel-using imports."""
    if os.getenv("DD_LLMOBS_ENABLED", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    if os.getenv("DD_TRACE_LLMOBS_IN_CODE", "1").strip().lower() in ("0", "false", "no"):
        return True  # ddtrace-run handles init externally
    try:
        from ddtrace.llmobs import LLMObs

        agentless = os.getenv("DD_LLMOBS_AGENTLESS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        integ = os.getenv("DD_LLMOBS_INTEGRATIONS_ENABLED", "true").strip().lower() not in ("0", "false", "no")
        LLMObs.enable(
            ml_app=os.getenv("DD_LLMOBS_ML_APP", "crew-streamlit"),
            api_key=os.getenv("DD_API_KEY"),
            site=os.getenv("DD_SITE", "datadoghq.com"),
            agentless_enabled=agentless,
            env=os.getenv("DD_ENV"),
            service=os.getenv("DD_SERVICE", "crew-streamlit"),
            integrations_enabled=integ,
        )
        return True
    except ModuleNotFoundError:
        return False


_DD_LLMOBS_ACTIVE = _init_datadog_llmobs()

import contextlib
import io
import platform
import sys
from typing import Any, Dict, Iterator

import streamlit as st


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@st.cache_resource
def get_langfuse() -> Any:
    from langfuse import Langfuse

    return Langfuse(
        public_key=_require_env("LANGFUSE_PUBLIC_KEY"),
        secret_key=_require_env("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


def _ensure_datadog_llmobs_enabled() -> bool:
    return _DD_LLMOBS_ACTIVE


@contextlib.contextmanager
def _datadog_workflow(
    question: str,
    agent_spec: Dict[str, Any],
    task_spec: Dict[str, Any],
    runtime: Dict[str, str],
) -> Iterator[None]:
    if not _ensure_datadog_llmobs_enabled():
        yield
        return
    try:
        from ddtrace.llmobs import LLMObs
    except ModuleNotFoundError:
        yield
        return

    wf_kwargs: Dict[str, Any] = {"name": "crewai.research"}
    session_id = os.getenv("DD_LLMOBS_SESSION_ID", "").strip()
    if session_id:
        wf_kwargs["session_id"] = session_id
    ml_app = os.getenv("DD_LLMOBS_ML_APP", "").strip()
    if ml_app:
        wf_kwargs["ml_app"] = ml_app

    with LLMObs.workflow(**wf_kwargs):
        LLMObs.annotate(
            input_data={
                "question": question,
                "agents": [agent_spec],
                "tasks": [task_spec],
                "runtime": runtime,
            },
            metadata={"framework": "crewai", "app": "streamlit", "observability": "datadog_llmobs"},
        )
        yield


@contextlib.contextmanager
def _datadog_kickoff_task() -> Iterator[None]:
    if not _ensure_datadog_llmobs_enabled():
        yield
        return
    try:
        from ddtrace.llmobs import LLMObs
    except ModuleNotFoundError:
        yield
        return

    with LLMObs.task(name="crew.kickoff"):
        yield


def run_research(question: str, langfuse: Any) -> Dict[str, str]:
    from crewai import Agent, Crew, Task

    agent_spec = {
        "role": "Researcher",
        "goal": "Research the user's question and answer clearly and accurately.",
        "backstory": "You are a diligent researcher who writes concise, well-structured answers with examples.",
        "verbose": True,
        "allow_delegation": False,
    }

    task_spec = {
        "description": f'Research the question: "{question}"',
        "expected_output": "A clear, concise answer with key points and 1-3 examples if applicable.",
    }

    researcher = Agent(
        role=agent_spec["role"],
        goal=agent_spec["goal"],
        backstory=agent_spec["backstory"],
        verbose=agent_spec["verbose"],
        allow_delegation=agent_spec["allow_delegation"],
    )

    task = Task(
        description=task_spec["description"],
        expected_output=task_spec["expected_output"],
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=True)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    with _datadog_workflow(question, agent_spec, task_spec, runtime):
        with langfuse.start_as_current_observation(
            name="crewai.research",
            as_type="chain",
            input={"question": question, "crew": {"agents": [agent_spec], "tasks": [task_spec]}, "runtime": runtime},
            metadata={"framework": "crewai", "app": "streamlit"},
        ) as root:
            with root.start_as_current_observation(
                name="crew.kickoff",
                as_type="span",
                input={"question": question, "agents": [agent_spec], "tasks": [task_spec]},
                metadata={"crew_verbose": True},
            ) as kickoff:
                with _datadog_kickoff_task():
                    try:
                        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                            result = crew.kickoff()
                        output = str(result)
                        kickoff.update(
                            output={"result": output, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}
                        )
                        root.update(output={"result": output})
                        if _ensure_datadog_llmobs_enabled():
                            try:
                                from ddtrace.llmobs import LLMObs

                                LLMObs.annotate(
                                    output_data={
                                        "result": output,
                                        "stdout": stdout_buf.getvalue(),
                                        "stderr": stderr_buf.getvalue(),
                                    },
                                    metadata={"crew_verbose": True},
                                )
                            except Exception:
                                pass
                        return {"result": output, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}
                    except Exception as e:
                        kickoff.update(
                            output={"error": repr(e), "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()},
                            level="ERROR",
                        )
                        root.update(output={"error": repr(e)}, level="ERROR")
                        if _ensure_datadog_llmobs_enabled():
                            try:
                                from ddtrace.llmobs import LLMObs

                                LLMObs.annotate(
                                    output_data={
                                        "error": repr(e),
                                        "stdout": stdout_buf.getvalue(),
                                        "stderr": stderr_buf.getvalue(),
                                    },
                                    metadata={"status": "error"},
                                )
                            except Exception:
                                pass
                        raise
                    finally:
                        try:
                            langfuse.flush()
                        except Exception:
                            pass
                        if _ensure_datadog_llmobs_enabled():
                            try:
                                from ddtrace.llmobs import LLMObs

                                LLMObs.flush()
                            except Exception:
                                pass


st.set_page_config(page_title="CrewAI + Langfuse Research", layout="wide")
st.title("CrewAI Researcher (traced to Langfuse)")

question = st.text_area("Research question", placeholder='e.g. "What is AI?"', height=120)
col1, col2 = st.columns([1, 3])

with col1:
    run_clicked = st.button("Run research", type="primary", disabled=not question.strip())

if run_clicked:
    try:
        langfuse = get_langfuse()
        with st.spinner("Running CrewAI..."):
            data = run_research(question.strip(), langfuse)
        st.subheader("Result")
        st.write(data["result"])

        with col2:
            with st.expander("Captured stdout/stderr (logged to Langfuse)", expanded=False):
                st.code(data["stdout"] or "", language="text")
                if data["stderr"]:
                    st.markdown("**stderr**")
                    st.code(data["stderr"], language="text")
    except Exception as e:
        st.error(f"Failed: {e}")

