from contextlib import contextmanager
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
        # start_as_current_observation sets the OTel current span, so nested
        # calls within this context automatically become child spans.
        with self._client.start_as_current_observation(**kwargs) as obs:
            if self._run_ctx is not None:
                try:
                    self._client.update_current_trace(
                        session_id=self._run_ctx.session_id or None,
                        user_id=self._run_ctx.user_id or None,
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
