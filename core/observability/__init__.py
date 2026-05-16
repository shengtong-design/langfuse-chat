from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from .base import BaseConnector, SpanHandle


class MultiSpanHandle(SpanHandle):
    def __init__(self, handles: list[SpanHandle]) -> None:
        self._handles = handles

    def set_output(self, output: Any) -> None:
        for h in self._handles:
            h.set_output(output)

    def mark_error(self) -> None:
        for h in self._handles:
            h.mark_error()


class ConnectorManager:
    """Fans every span operation out to all enabled connectors.

    To add a new connector: create a class in observability/ that implements
    BaseConnector, then pass an instance to ConnectorManager in crew_app.py.
    """

    def __init__(self, connectors: list[BaseConnector]) -> None:
        self._connectors = [c for c in connectors if c.enabled]

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[MultiSpanHandle]:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(c.span(name, span_type, input_data, metadata))
                for c in self._connectors
            ]
            yield MultiSpanHandle(handles)

    def flush(self) -> None:
        for c in self._connectors:
            c.flush()

    def update_run_context(self, context: Any) -> None:
        for c in self._connectors:
            c.update_run_context(context)

    def for_callbacks(self) -> "ConnectorManager":
        """Return a manager containing only connectors that handle step callbacks.

        Connectors with native CrewAI instrumentation (e.g. Datadog) set
        handles_step_callbacks=False so ddtrace's own patching runs unobstructed.
        """
        return ConnectorManager([c for c in self._connectors if c.handles_step_callbacks])
