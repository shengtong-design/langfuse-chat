"""Shared utilities for crew execution."""

import contextlib
import io
import logging
from typing import Any

log = logging.getLogger(__name__)


def kickoff_crew(
    crew: Any,
    obs: Any,
    input_data: dict[str, Any] | None = None,
) -> tuple[Any, str, str]:
    """Run crew.kickoff(inputs=...) inside a span with captured stdout/stderr.

    Returns (CrewOutput, stdout, stderr). Raises on failure after updating the
    span with error context so the exception still propagates to the caller.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with obs.span("crew.kickoff", "span", input_data=input_data) as kickoff:
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                result = crew.kickoff(inputs=input_data or {})
            stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            kickoff.set_output({"result": str(result), "stdout": stdout, "stderr": stderr})
            return result, stdout, stderr
        except Exception as e:
            stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            log.warning("crew.kickoff failed", exc_info=True)
            kickoff.set_output({"error": repr(e), "stdout": stdout, "stderr": stderr})
            kickoff.mark_error()
            raise


def extract_question(input_item: Any) -> str:
    """Normalise a Langfuse dataset item input to a plain question string."""
    if isinstance(input_item, dict):
        value = input_item.get("question") or input_item.get("query") or input_item.get("input")
        if value is None:
            log.warning(
                "extract_question: no 'question'/'query'/'input' key found; serialising dict %r",
                input_item,
            )
            return str(input_item)
        return str(value)
    return str(input_item)
