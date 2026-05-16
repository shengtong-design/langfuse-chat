import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .base import BaseConnector, NullSpanHandle, SpanHandle

log = logging.getLogger(__name__)

_TYPE_TO_DD = {
    "chain": "workflow",
    "span": "task",
    "agent": "agent",
    "tool": "tool",
    "generation": "llm",
}


class DatadogSpanHandle(SpanHandle):
    def __init__(self) -> None:
        self._output: Any | None = None
        self._error: bool = False

    def set_output(self, output: Any) -> None:
        self._output = output

    def mark_error(self) -> None:
        self._error = True


class DatadogConnector(BaseConnector):
    # ddtrace patches CrewAI natively — our step/task callbacks would double-instrument it.
    handles_step_callbacks: bool = False

    def __init__(self, active: bool) -> None:
        self._active = active
        self._run_ctx: Any | None = None

    def update_run_context(self, context: Any) -> None:
        self._run_ctx = context

    @property
    def enabled(self) -> bool:
        return self._active

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        if not self._active:
            yield NullSpanHandle()
            return

        try:
            from ddtrace.llmobs import LLMObs

            dd_type = _TYPE_TO_DD.get(span_type, "task")
            method = getattr(LLMObs, dd_type, None)
        except (ModuleNotFoundError, AttributeError):
            log.info("Datadog LLMObs unavailable — spans will be no-ops", exc_info=True)
            yield NullSpanHandle()
            return

        if method is None:
            log.info("Datadog LLMObs has no method for span type %r", span_type)
            yield NullSpanHandle()
            return

        span_kwargs: dict[str, Any] = {"name": name}
        session_id = (self._run_ctx.session_id if self._run_ctx else "") or os.getenv(
            "DD_LLMOBS_SESSION_ID", ""
        ).strip()
        if session_id:
            span_kwargs["session_id"] = session_id

        handle = DatadogSpanHandle()
        with method(**span_kwargs):
            merged_metadata = dict(metadata or {})
            if self._run_ctx is not None:
                # Merge RunContext into metadata so all fields appear in span detail view.
                # Tags are set separately for Datadog filtering/grouping.
                merged_metadata.update(self._run_ctx.as_metadata())
            annotate_kwargs: dict[str, Any] = {"metadata": merged_metadata}
            if input_data:
                annotate_kwargs["input_data"] = input_data
            if self._run_ctx is not None:
                dd_tags = self._run_ctx.as_dd_tags()
                if dd_tags:
                    annotate_kwargs["tags"] = dd_tags
            LLMObs.annotate(**annotate_kwargs)
            try:
                yield handle
            finally:
                if handle._output is not None:
                    LLMObs.annotate(output_data=handle._output)
                if handle._error:
                    LLMObs.annotate(metadata={"error": True})

    def flush(self) -> None:
        if not self._active:
            return
        try:
            from ddtrace.llmobs import LLMObs

            LLMObs.flush()
        except Exception:
            log.warning("Datadog LLMObs flush failed", exc_info=True)
