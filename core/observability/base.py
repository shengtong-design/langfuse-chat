from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


class SpanHandle(ABC):
    @abstractmethod
    def update(self, output: Any = None, level: str = "DEFAULT") -> None: ...


class NullSpanHandle(SpanHandle):
    def update(self, output: Any = None, level: str = "DEFAULT") -> None:
        pass


class BaseConnector(ABC):
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
