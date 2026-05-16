from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol


class ObsManager(Protocol):
    """Structural type for the observability manager passed to crew.run().

    Satisfied by ConnectorManager and EnrichedConnectorManager without
    inheritance — crew files stay decoupled from the observability package.
    """

    def span(
        self,
        name: str,
        span_type: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def flush(self) -> None: ...

    def update_run_context(self, context: Any) -> None: ...


class SpanHandle(ABC):
    """Handle returned by connector.span() context managers.

    Callers may call set_output() and/or mark_error() at most once per span,
    before the context manager exits. Connectors do not merge multiple calls.
    """

    @abstractmethod
    def set_output(self, output: Any) -> None: ...

    @abstractmethod
    def mark_error(self) -> None: ...


class NullSpanHandle(SpanHandle):
    def set_output(self, output: Any) -> None:
        pass

    def mark_error(self) -> None:
        pass


class BaseConnector(ABC):
    # Set to False on connectors with native CrewAI instrumentation (e.g. Datadog/ddtrace)
    # so their own patching runs unobstructed by our step/task callbacks.
    handles_step_callbacks: bool = True

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @abstractmethod
    @contextmanager
    def span(
        self,
        name: str,
        span_type: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[SpanHandle]: ...

    def flush(self) -> None:
        pass

    def update_run_context(self, context: Any) -> None:
        pass
