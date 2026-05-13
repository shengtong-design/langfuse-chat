from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class RunContext:
    """Carries per-run identity/environment metadata propagated to every span."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    environment: str = "dev"
    app_version: str = "1.0.0"
    crew_name: str = ""

    def as_metadata(self) -> dict:
        return {k: v for k, v in {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "environment": self.environment,
            "app_version": self.app_version,
            "crew_name": self.crew_name,
        }.items() if v}

    def as_tags(self) -> list:
        """Langfuse-style tags: list of 'key:value' strings."""
        tags = []
        if self.environment:
            tags.append(f"env:{self.environment}")
        if self.crew_name:
            tags.append(f"crew:{self.crew_name}")
        if self.app_version:
            tags.append(f"version:{self.app_version}")
        return tags

    def as_dd_tags(self) -> dict:
        """Datadog-style tags: {'key': 'value'} dict for LLMObs.annotate(tags=...)."""
        tags = {}
        if self.environment:
            tags["env"] = self.environment
        if self.crew_name:
            tags["crew"] = self.crew_name
        if self.app_version:
            tags["version"] = self.app_version
        return tags
