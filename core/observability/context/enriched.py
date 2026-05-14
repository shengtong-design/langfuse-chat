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
    def __init__(self, base: "ConnectorManager", context: RunContext) -> None:
        self._base = base
        self._ctx = context
        # Build a callbacks-only manager by filtering connectors directly.
        # Avoids calling base.for_callbacks() which would fail on a stale
        # @st.cache_resource instance whose class predates that method.
        raw = getattr(base, '_connectors', [])
        cb_connectors = [c for c in raw if getattr(c, 'handles_step_callbacks', True)]
        cb_base = type(base)(cb_connectors)
        self._callbacks = CrewCallbacks(EnrichedConnectorManager._from_parts(cb_base, context))
        base.update_run_context(context)

    @classmethod
    def _from_parts(cls, base: "ConnectorManager", context: RunContext) -> "EnrichedConnectorManager":
        """Create a bare instance with no callbacks wiring (used internally)."""
        inst = object.__new__(cls)
        inst._base = base
        inst._ctx = context
        inst._callbacks = None  # type: ignore[assignment]
        return inst

    @property
    def crew_callbacks(self) -> CrewCallbacks:
        return self._callbacks

    def _merged_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # Emit in reverse-alphabetical insertion order so the Langfuse UI, which
        # renders dict keys in reverse-insertion order, displays them forward
        # alphabetically. This couples us to a Langfuse rendering quirk — if
        # they ever switch to insertion-order display, flip `reverse` here.
        merged = self._ctx.as_metadata()
        if metadata:
            merged.update(metadata)
        return dict(sorted(merged.items(), reverse=True))

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
