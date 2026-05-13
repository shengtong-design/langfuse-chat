"""
ADDON: Runtime Prompt Management
=================================
Loads agent prompts from Langfuse at runtime with YAML fallback.

Separation of concerns (per architecture guideline):
  GitHub  — YAML fallback = deterministic default contract
  Langfuse — runtime prompt = rapidly evolving AI behavior

Usage in a crew:
    from core.prompts import PromptLoader

    loader = PromptLoader()
    prompt = loader.get(
        "researcher_agent",            # name in Langfuse
        fallback=YAML_AGENT_DEFAULTS,  # GitHub fallback
        label="production",
    )
    agent = Agent(
        role=prompt.config["role"],
        goal=prompt.config["goal"],
        backstory=prompt.config["backstory"],
        ...
    )
    # Pass prompt.as_metadata() into your root span so version is tracked
    # in Langfuse traces and Datadog.

Langfuse prompt setup:
  - Create a prompt in Langfuse UI (type: "chat" or "text")
  - Add a Config dict with keys: role, goal, backstory (match your YAML keys)
  - Label it "production" to promote it; "staging" for experiments
  - Langfuse config keys override the YAML fallback; missing keys keep defaults
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Loaded prompt config plus version metadata for span tracing."""

    config: Dict[str, Any]
    version: str = "fallback"
    name: str = ""
    label: str = ""

    def as_metadata(self) -> Dict[str, Any]:
        """Returns a flat dict suitable for Langfuse / Datadog span metadata."""
        return {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_label": self.label,
        }


class PromptLoader:
    """Fetches prompts from Langfuse; falls back to a provided dict on any failure.

    The Langfuse prompt `config` dict is merged over the fallback, so YAML
    defaults remain active for any key that Langfuse does not override.

    Resilience:
      - Langfuse unreachable      → uses fallback silently
      - Prompt label not found    → uses fallback with a warning
      - Env vars not set          → uses fallback silently
    """

    def __init__(self) -> None:
        self._client: Optional[Any] = None

    def _client_or_none(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        if not (pk and sk):
            return None
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=pk,
                secret_key=sk,
                base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
            )
        except Exception:
            log.debug("PromptLoader: Langfuse client init failed", exc_info=True)
        return self._client

    def get(
        self,
        name: str,
        fallback: Dict[str, Any],
        label: str = "production",
        cache_ttl: int = 300,
    ) -> PromptResult:
        """Fetch prompt from Langfuse and merge with fallback.

        Args:
            name:      Prompt name in Langfuse (e.g. "researcher_agent").
            fallback:  Dict of agent fields from YAML — used when Langfuse
                       is unavailable or the prompt does not exist yet.
            label:     Langfuse rollout label ("production", "staging", ...).
            cache_ttl: Seconds the SDK caches the prompt locally (default 5 min).

        Returns:
            PromptResult with merged config and version metadata.
        """
        client = self._client_or_none()
        if client is not None:
            try:
                prompt = client.get_prompt(name, label=label, cache_ttl_seconds=cache_ttl)
                config = {**fallback, **dict(prompt.config)}
                log.info("Loaded prompt '%s' version=%s label=%s", name, prompt.version, label)
                return PromptResult(
                    config=config,
                    version=str(prompt.version),
                    name=name,
                    label=label,
                )
            except Exception:
                log.warning(
                    "Langfuse prompt '%s' (label=%s) unavailable — using YAML fallback",
                    name,
                    label,
                    exc_info=True,
                )
        return PromptResult(config=dict(fallback), version="fallback", name=name, label=label)
