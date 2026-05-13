"""Shared utilities for crew execution."""
import contextlib
import io
import logging
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)


def kickoff_crew(
    crew: Any,
    obs: Any,
    input_data: Dict[str, Any] = None,
) -> Tuple[Any, str, str]:
    """Run crew.kickoff() inside a crew.kickoff span with captured stdout/stderr.

    Returns (CrewOutput, stdout, stderr). Raises on failure after updating the
    span with error context so the exception still propagates to the caller.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with obs.span("crew.kickoff", "span", input_data=input_data) as kickoff:
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                result = crew.kickoff()
            stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            kickoff.update(output={"result": str(result), "stdout": stdout, "stderr": stderr})
            return result, stdout, stderr
        except Exception as e:
            stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            log.debug("crew.kickoff failed", exc_info=True)
            kickoff.update(
                output={"error": repr(e), "stdout": stdout, "stderr": stderr},
                level="ERROR",
            )
            raise


def extract_question(input_item: Any) -> str:
    """Normalise a Langfuse dataset item input to a plain question string."""
    if isinstance(input_item, dict):
        return str(
            input_item.get("question")
            or input_item.get("query")
            or input_item.get("input")
            or input_item
        )
    return str(input_item)
