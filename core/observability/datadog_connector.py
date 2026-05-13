import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

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
        self._output: Optional[Any] = None
        self._error: bool = False

    def update(self, output: Any = None, level: str = "DEFAULT") -> None:
        self._output = output
        if level == "ERROR":
            self._error = True


class DatadogConnector(BaseConnector):
    def __init__(self, active: bool) -> None:
        self._active = active
        self._run_ctx: Optional[Any] = None

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
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanHandle]:
        if not self._active:
            yield NullSpanHandle()
            return

        try:
            from ddtrace.llmobs import LLMObs
            dd_type = _TYPE_TO_DD.get(span_type, "task")
            method = getattr(LLMObs, dd_type, None)
        except (ModuleNotFoundError, AttributeError):
            log.debug("Datadog LLMObs unavailable", exc_info=True)
            yield NullSpanHandle()
            return

        if method is None:
            log.debug("Datadog LLMObs has no method for span type %r", span_type)
            yield NullSpanHandle()
            return

        span_kwargs: Dict[str, Any] = {"name": name}
        session_id = (
            (self._run_ctx.session_id if self._run_ctx else "")
            or os.getenv("DD_LLMOBS_SESSION_ID", "").strip()
        )
        if session_id:
            span_kwargs["session_id"] = session_id

        handle = DatadogSpanHandle()
        with method(**span_kwargs):
            annotate_kwargs: Dict[str, Any] = {"metadata": metadata or {}}
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
            log.debug("Datadog LLMObs flush failed", exc_info=True)
