"""EvalOps Streamlit tab — Experiment Run Request form.

Replaces the previous `tab_experiment` block in `crew_app.py`.
Delegates to `evalops.runners.pipeline.run_pipeline` so the CLI and
UI share one implementation.
"""

from __future__ import annotations

import streamlit as st

from evalops.metric_config import REGISTRY as METRIC_REGISTRY
from evalops.runners.pipeline import PipelineConfig, run_pipeline


def render() -> None:
    st.caption(
        "Run an EvalOps experiment: dataset + crew + prompt label → "
        "Langfuse LLM-as-a-Judge scores → local Markdown report."
    )

    if "evalops_running" not in st.session_state:
        st.session_state.evalops_running = False

    available_metrics = sorted(METRIC_REGISTRY.keys())
    default_metrics = [m for m in ("Conciseness", "Hallucination", "Correctness") if m in available_metrics]

    with st.form("evalops_form"):
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.text_input("Dataset name", value="crew-research-eval")
            crew = st.selectbox("Crew", ["researcher"], index=0)
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

        cfg = PipelineConfig(
            dataset=dataset.strip(),
            crew=crew,
            prompt_label=prompt_label,
            metrics=list(metrics),
            wait_seconds=int(wait_seconds),
            experiment_name=(experiment_name.strip() or None),
        )

        try:
            st.session_state.evalops_running = True
            with st.spinner(
                f"Running '{cfg.dataset}' on {cfg.crew} (wait {cfg.wait_seconds}s for judges)..."
            ):
                result = run_pipeline(cfg)
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
