from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Protocol


class ObsManager(Protocol):
    """Structural type for the observability manager passed to crew.run().

    Satisfied by ConnectorManager and EnrichedConnectorManager without
    inheritance — crew files stay decoupled from the observability package.
    """

    def span(
        self,
        name: str,
        span_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any: ...

    def flush(self) -> None: ...

    def update_run_context(self, context: Any) -> None: ...


class SpanHandle(ABC):
    @abstractmethod
    def update(self, output: Any = None, level: str = "DEFAULT") -> None: ...


class NullSpanHandle(SpanHandle):
    def update(self, output: Any = None, level: str = "DEFAULT") -> None:
        pass


class BaseConnector(ABC):
    # Set to False on connectors that have native CrewAI instrumentation (e.g. Datadog/ddtrace).
    # When False, the connector is excluded from CrewCallbacks so its own patching runs unobstructed.
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
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanHandle]: ...

    def flush(self) -> None:
        pass

    def update_run_context(self, context: Any) -> None:
        pass
