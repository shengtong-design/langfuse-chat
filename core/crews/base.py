from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from core.observability import ConnectorManager


class BaseCrew(ABC):
    """All crews implement this interface.

    To add a new crew: subclass BaseCrew, implement crew_name and run(),
    then register it in crews/__init__.py CREWS dict.
    """

    @property
    @abstractmethod
    def crew_name(self) -> str: ...

    @abstractmethod
    def run(self, inputs: Dict[str, Any], obs: "ConnectorManager") -> Dict[str, Any]:
        """Execute the crew and return a dict with at least a 'result' key."""
        ...
