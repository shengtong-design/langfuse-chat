"""
ADDON: EnrichedConnectorManager
================================
Wraps ConnectorManager to automatically:
  1. Inject RunContext (session_id, run_id, user_id, environment, ...) into
     every span's metadata.
  2. Push the context to each connector via update_run_context() so backends
     can set trace-level fields (e.g. Langfuse session/user, Datadog session_id).
  3. Expose crew_callbacks for wiring CrewAI step/task callbacks.

Usage in crew_app.py:

    from core.observability.context import make_run_context, EnrichedConnectorManager

    obs = EnrichedConnectorManager(_get_connectors(), make_run_context("researcher"))
    crew.run(inputs, obs)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

if TYPE_CHECKING:
    # Runtime imports of core.observability here would re-enter the core package
    # while crew_app.py is still loading it, causing KeyError: 'core' on hot-reload.
    # These are type-hint-only; from __future__ import annotations keeps them lazy.
    from core.observability import ConnectorManager
    from core.observability.base import SpanHandle

from .callbacks import CrewCallbacks
from .run_context import RunContext


class EnrichedConnectorManager:
    def __init__(self, base: ConnectorManager, context: RunContext) -> None:
        self._base = base
        self._ctx = context
        self._callbacks = CrewCallbacks(self)
        base.update_run_context(context)

    @property
    def crew_callbacks(self) -> CrewCallbacks:
        return self._callbacks

    def _merged_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = self._ctx.as_metadata()
        if metadata:
            merged.update(metadata)
        return merged

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanHandle]:
        with self._base.span(name, span_type, input_data, self._merged_metadata(metadata)) as handle:
            yield handle

    def flush(self) -> None:
        self._base.flush()

    def update_run_context(self, context: Any) -> None:
        self._base.update_run_context(context)
