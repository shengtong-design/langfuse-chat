from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, Optional

from .base import BaseConnector, SpanHandle


class LangfuseSpanHandle(SpanHandle):
    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def update(self, output: Any = None, level: str = "DEFAULT") -> None:
        kwargs: Dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if level != "DEFAULT":
            kwargs["level"] = level
        if kwargs:
            self._obs.update(**kwargs)


class LangfuseConnector(BaseConnector):
    def __init__(self, client: Any) -> None:
        self._client = client
        self._run_ctx: Optional[Any] = None

    def update_run_context(self, context: Any) -> None:
        self._run_ctx = context

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[LangfuseSpanHandle]:
        kwargs: Dict[str, Any] = {"name": name, "as_type": span_type}
        if input_data is not None:
            kwargs["input"] = input_data
        if metadata is not None:
            kwargs["metadata"] = metadata
        # propagate_attributes MUST wrap start_as_current_observation — Langfuse
        # reads session_id/user_id from the OTel context at trace creation time.
        # Calling update_current_trace() after the span opens is too late for
        # session grouping (it updates the object but not the session index).
        with ExitStack() as stack:
            if self._run_ctx is not None:
                try:
                    from langfuse import propagate_attributes
                    attrs: Dict[str, Any] = {}
                    if self._run_ctx.session_id:
                        attrs["session_id"] = self._run_ctx.session_id
                    if self._run_ctx.user_id:
                        attrs["user_id"] = self._run_ctx.user_id
                    if attrs:
                        stack.enter_context(propagate_attributes(**attrs))
                except Exception:
                    pass
            obs = stack.enter_context(self._client.start_as_current_observation(**kwargs))
            if self._run_ctx is not None:
                try:
                    # tags are trace-level metadata, not session routing — safe to
                    # set after span opens.
                    self._client.update_current_trace(
                        tags=self._run_ctx.as_tags() or None,
                    )
                except Exception:
                    pass
            yield LangfuseSpanHandle(obs)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            pass
