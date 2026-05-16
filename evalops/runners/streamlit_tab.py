"""EvalOps Streamlit tab — Experiment Run Request form.

Replaces the previous `tab_experiment` block in `crew_app.py`.
Delegates to `evalops.runners.pipeline.run_pipeline` so the CLI and
UI share one implementation.

The "last run" result is persisted in `st.session_state` so the
PDF/Markdown download buttons remain available across Streamlit
reruns until the tab is replaced or the session ends.

Pipeline imports are deferred to render-time because pipeline.py
transitively pulls in `core.prompts.loader` / `flows.research_flow`,
and on Streamlit Cloud crew_app.py is mid-init when this module
imports — ddtrace's import wrapper has not yet been installed at
that point, and pulling those modules in too early causes a
KeyError in importlib._find_and_load.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from evalops.metric_config import REGISTRY as METRIC_REGISTRY

if TYPE_CHECKING:
    from evalops.runners.pipeline import PipelineResult


def _md_to_pdf_bytes(md_text: str) -> bytes:
    """Render Markdown to PDF bytes. Returns b'' on failure (UI shows MD fallback)."""
    try:
        from evalops.pdf_export import md_to_pdf_bytes
    except ImportError:
        return b""
    try:
        return md_to_pdf_bytes(md_text)
    except Exception:
        return b""


def _render_download_panel(result: PipelineResult) -> None:
    """Render persistent download buttons for the most recent run."""
    try:
        md_text = result.report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.info("Last report file is no longer on disk.")
        return

    st.markdown(f"**Last run:** `{result.experiment_name}` — {result.score_count} scores across {result.distinct_evaluators} evaluator(s)")

    col_pdf, col_md = st.columns(2)
    with col_pdf:
        pdf_bytes = _md_to_pdf_bytes(md_text)
        if pdf_bytes:
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{result.experiment_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF unavailable (renderer error or missing dep).")
    with col_md:
        st.download_button(
            label="Download Markdown",
            data=md_text,
            file_name=f"{result.experiment_name}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render() -> None:
    st.caption(
        "Run an EvalOps experiment: dataset + flow + prompt label → "
        "Langfuse LLM-as-a-Judge scores → local Markdown report."
    )

    if "evalops_running" not in st.session_state:
        st.session_state.evalops_running = False

    last_result: PipelineResult | None = st.session_state.get("evalops_last_result")
    if last_result is not None:
        _render_download_panel(last_result)
        st.divider()

    available_metrics = sorted(METRIC_REGISTRY.keys())
    default_metrics = [m for m in ("Conciseness", "Hallucination", "Correctness") if m in available_metrics]

    with st.form("evalops_form"):
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.text_input("Dataset name", value="crew-research-eval")
            flow = st.selectbox("Flow", ["researcher"], index=0)
            prompt_label = st.selectbox(
                "Prompt label",
                ["production", "staging", "candidate"],
                index=0,
            )
        with col2:
            metrics = st.multiselect("Metrics", available_metrics, default=default_metrics)
            wait_seconds = st.slider("Judge wait (seconds)", 0, 300, 180, step=30)
            experiment_name = st.text_input("Experiment name (optional)")

        submit = st.form_submit_button(
            "Running..." if st.session_state.evalops_running else "Run evaluation",
            type="primary",
            disabled=st.session_state.evalops_running,
        )

    if submit and not st.session_state.evalops_running:
        if not dataset.strip():
            st.warning("Please enter a dataset name.")
            return
        if not metrics:
            st.warning("Please select at least one metric.")
            return

        from evalops.runners.pipeline import PipelineConfig, run_pipeline

        cfg = PipelineConfig(
            dataset=dataset.strip(),
            crew=flow,
            prompt_label=prompt_label,
            metrics=list(metrics),
            wait_seconds=int(wait_seconds),
            experiment_name=(experiment_name.strip() or None),
        )

        try:
            st.session_state.evalops_running = True
            with st.spinner(
                f"Running flow '{cfg.crew}' on dataset '{cfg.dataset}' "
                f"(wait {cfg.wait_seconds}s for judges)..."
            ):
                result = run_pipeline(cfg)
            st.session_state["evalops_last_result"] = result
            st.success(f"Experiment **{result.experiment_name}** complete.")
            st.markdown(f"- Items: **{result.item_count}**")
            st.markdown(
                f"- Scores: **{result.score_count}** across "
                f"**{result.distinct_evaluators}** evaluator(s)"
            )
            st.markdown(f"- Manifest: `{result.manifest_path}`")
            st.markdown(f"- Report: `{result.report_path}`")
            st.caption(f"Check Langfuse → Datasets → `{cfg.dataset}` → Experiments")
            try:
                with st.expander("Show report"):
                    st.markdown(result.report_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        except Exception as e:
            st.error(f"Failed: {e}")
        finally:
            st.session_state.evalops_running = False
