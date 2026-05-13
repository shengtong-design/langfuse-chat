from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseConnector, SpanHandle


class MultiSpanHandle(SpanHandle):
    def __init__(self, handles: List[SpanHandle]) -> None:
        self._handles = handles

    def update(self, output: Any = None, level: str = "DEFAULT") -> None:
        for h in self._handles:
            h.update(output=output, level=level)


class ConnectorManager:
    """Fans every span operation out to all enabled connectors.

    To add a new connector: create a class in observability/ that implements
    BaseConnector, then pass an instance to ConnectorManager in crew_app.py.
    """

    def __init__(self, connectors: List[BaseConnector]) -> None:
        self._connectors = [c for c in connectors if c.enabled]

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
