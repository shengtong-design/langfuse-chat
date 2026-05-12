import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from .base import BaseConnector, NullSpanHandle, SpanHandle

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

    def update(self, output: Any = None, level: str = "DEFAULT") -> None:
        self._output = output


class DatadogConnector(BaseConnector):
    def __init__(self, active: bool) -> None:
        self._active = active

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

        # Separate import errors from crew errors — a broken import falls back
        # to NullSpanHandle; crew exceptions propagate normally through yield.
        try:
            from ddtrace.llmobs import LLMObs
            dd_type = _TYPE_TO_DD.get(span_type, "task")
            method = getattr(LLMObs, dd_type, None)
        except (ModuleNotFoundError, AttributeError):
            yield NullSpanHandle()
            return

        if method is None:
            yield NullSpanHandle()
            return

        span_kwargs: Dict[str, Any] = {"name": name}
        session_id = os.getenv("DD_LLMOBS_SESSION_ID", "").strip()
        if session_id:
            span_kwargs["session_id"] = session_id

        handle = DatadogSpanHandle()
        with method(**span_kwargs):
            if input_data:
                LLMObs.annotate(input_data=input_data, metadata=metadata or {})
            try:
                yield handle
            finally:
                # Annotate output while still inside this span's context so
                # LLMObs.annotate() targets the correct nesting level.
                if handle._output is not None:
                    LLMObs.annotate(output_data=handle._output)

    def flush(self) -> None:
        if not self._active:
            return
        try:
            from ddtrace.llmobs import LLMObs
            LLMObs.flush()
        except Exception:
            pass
