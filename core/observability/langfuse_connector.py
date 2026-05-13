import logging
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, Optional

from .base import BaseConnector, SpanHandle

log = logging.getLogger(__name__)


class LangfuseSpanHandle(SpanHandle):
    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def set_output(self, output: Any) -> None:
        self._obs.update(output=output)

    def mark_error(self) -> None:
        self._obs.update(level="ERROR")


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

        # Open the trace first, then call propagate_attributes inside it.
        # The docs pattern is: trace root open → propagate_attributes inside →
        # which updates the current trace with session_id/user_id and propagates
        # them to child spans via OTel baggage.
        with ExitStack() as stack:
            obs = stack.enter_context(self._client.start_as_current_observation(**kwargs))
            if self._run_ctx is not None:
                try:
                    from langfuse import propagate_attributes
                    attrs: Dict[str, Any] = {}
                    if self._run_ctx.session_id:
                        attrs["session_id"] = self._run_ctx.session_id
                    if self._run_ctx.user_id:
                        attrs["user_id"] = self._run_ctx.user_id
                    tags = self._run_ctx.as_tags()
                    if tags:
                        attrs["tags"] = tags
                    if attrs:
                        stack.enter_context(propagate_attributes(**attrs))
                except Exception:
                    log.debug("Langfuse context propagation failed", exc_info=True)
            yield LangfuseSpanHandle(obs)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            log.debug("Langfuse flush failed", exc_info=True)
