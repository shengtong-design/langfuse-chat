from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class RunContext:
    """Carries per-run identity and environment metadata propagated to every span."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    environment: str = "dev"
    app_version: str = "1.0.0"
    crew_name: str = ""
    flow_name: str = ""
    deployment_sha: str = ""
    crew_version: str = ""
    flow_version: str = ""
    model_version: str = ""
    _workflow_id: str = field(default="", repr=False)

    @property
    def workflow_id(self) -> str:
        """Defaults to run_id when not explicitly set."""
        return self._workflow_id or self.run_id

    @workflow_id.setter
    def workflow_id(self, value: str) -> None:
        self._workflow_id = value

    def as_metadata(self) -> dict:
        return {k: v for k, v in {
            "session_id":     self.session_id,
            "run_id":         self.run_id,
            "user_id":        self.user_id,
            "environment":    self.environment,
            "app_version":    self.app_version,
            "crew_name":      self.crew_name,
            "flow_name":      self.flow_name,
            "deployment_sha": self.deployment_sha,
            "crew_version":   self.crew_version,
            "flow_version":   self.flow_version,
            "model_version":  self.model_version,
            "workflow_id":    self.workflow_id,
        }.items() if v}

    def as_tags(self) -> list:
        tags = []
        if self.environment:
            tags.append(f"env:{self.environment}")
        if self.crew_name:
            tags.append(f"crew:{self.crew_name}")
        if self.flow_name:
            tags.append(f"flow:{self.flow_name}")
        if self.app_version:
            tags.append(f"version:{self.app_version}")
        if self.deployment_sha:
            tags.append(f"sha:{self.deployment_sha[:8]}")
        return tags

    def as_dd_tags(self) -> dict:
        return {k: v for k, v in {
            "env":            self.environment,
            "crew":           self.crew_name,
            "flow":           self.flow_name,
            "version":        self.app_version,
            "deployment_sha": self.deployment_sha[:8] if self.deployment_sha else "",
            "crew_version":   self.crew_version,
            "flow_version":   self.flow_version,
            "model_version":  self.model_version,
            "workflow_id":    self.workflow_id,
            "session_id":     self.session_id,
        }.items() if v}
