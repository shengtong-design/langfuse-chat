from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class RunContext:
    """Carries per-run identity and environment metadata propagated to every span.

    Required by guideline section 12:
      prompt_version, agent_version, crew_version, model_version,
      workflow_id, deployment_sha, environment
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    environment: str = "dev"
    app_version: str = "1.0.0"
    crew_name: str = ""
    # Extended metadata (guideline section 12)
    deployment_sha: str = ""
    crew_version: str = ""
    model_version: str = ""
    workflow_id: str = ""   # defaults to run_id when not set by a flow

    def as_metadata(self) -> dict:
        return {k: v for k, v in {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "environment": self.environment,
            "app_version": self.app_version,
            "crew_name": self.crew_name,
            "deployment_sha": self.deployment_sha,
            "crew_version": self.crew_version,
            "model_version": self.model_version,
            "workflow_id": self.workflow_id or self.run_id,
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
        if self.deployment_sha:
            tags.append(f"sha:{self.deployment_sha[:8]}")
        return tags

    def as_dd_tags(self) -> dict:
        """Datadog-style tags: {'key': 'value'} dict for LLMObs.annotate(tags=...).

        Tags are indexed in Datadog and usable for filtering/grouping.
        All RunContext fields are included so native ddtrace CrewAI spans
        that share the same trace inherit them via the root span context.
        """
        return {k: v for k, v in {
            "env":            self.environment,
            "crew":           self.crew_name,
            "version":        self.app_version,
            "deployment_sha": self.deployment_sha[:8] if self.deployment_sha else "",
            "crew_version":   self.crew_version,
            "model_version":  self.model_version,
            "workflow_id":    self.workflow_id or self.run_id,
            "session_id":     self.session_id,
        }.items() if v}
