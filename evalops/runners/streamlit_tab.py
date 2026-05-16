"""EvalOps Streamlit tab — Experiment Run Request form.

Replaces the previous `tab_experiment` block in `crew_app.py`.
Delegates to `evalops.runners.pipeline.run_pipeline` so the CLI and
UI share one implementation.

The "last run" result is persisted in `st.session_state` so the
PDF/Markdown download buttons remain available across Streamlit
reruns until the tab is replaced or the session ends.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from evalops.metric_config import REGISTRY as METRIC_REGISTRY
from evalops.runners.pipeline import PipelineConfig, PipelineResult, run_pipeline

_PDF_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; color: #222; }
h1 { font-size: 16pt; }
h2 { font-size: 12pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 14pt; }
h3 { font-size: 10pt; }
table { border-collapse: collapse; margin: 6pt 0; }
th, td { border: 1px solid #aaa; padding: 3pt 5pt; font-size: 8pt; }
th { background-color: #eee; }
code, pre { font-family: 'Courier New', monospace; font-size: 8pt; background: #f5f5f5; padding: 2pt; }
pre { padding: 6pt; white-space: pre-wrap; }
"""


def _md_to_pdf_bytes(md_text: str) -> bytes:
    """Render Markdown to PDF bytes. Returns b'' on failure (UI shows MD fallback)."""
    try:
        import markdown
        from xhtml2pdf import pisa
    except ImportError:
        return b""
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_doc = f"<html><head><style>{_PDF_CSS}</style></head><body>{html_body}</body></html>"
    buf = io.BytesIO()
    result = pisa.CreatePDF(src=html_doc, dest=buf, encoding="utf-8")
    if result.err:
        return b""
    return buf.getvalue()


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
        "Run an EvalOps experiment: dataset + crew + prompt label → "
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
