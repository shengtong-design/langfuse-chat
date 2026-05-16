"""
ADDON: Runtime Prompt Management
=================================
Loads CrewAI concept prompts (agents, tasks, ...) from Langfuse at runtime
with YAML fallback.

Separation of concerns (per architecture guideline):
  GitHub  — YAML fallback = deterministic default contract
  Langfuse — runtime prompt = rapidly evolving AI behavior

Usage in a crew (the per-concept caller in crews/base.py prepends the
namespace prefix; this loader stays concept-agnostic):

    from core.prompts import PromptLoader

    loader = PromptLoader()
    prompt = loader.get(
        "agent.researcher",            # namespaced name in Langfuse
        fallback=YAML_AGENT_DEFAULTS,  # GitHub fallback
        label="production",
    )

Langfuse prompt setup:
  - Create a prompt in Langfuse UI (type: "chat" or "text").
  - Name it with the CrewAI concept prefix so agent and task prompts can
    never collide on the same key:
      * Agents → "agent.<name>"   e.g. "agent.researcher"
      * Tasks  → "task.<name>"    e.g. "task.research_task"
    YAML keys stay bare (both concepts use `prompt_key`: e.g.
    `prompt_key: researcher` for agents, `prompt_key: research_task` for tasks);
    the prefix is added by the per-concept loader. Extend the same pattern
    for future concepts (crew., tool., knowledge.).
  - Add a Config dict:
      * Agents → role, goal, backstory  (match your YAML fallback keys)
      * Tasks  → description, expected_output
  - Label it "production" to promote it; "staging" for experiments.
  - Langfuse config keys override the YAML fallback. For tasks, only the LLM-
    text fields are pulled, so wiring (agent, context, tools, ...) cannot be
    changed via a Langfuse edit.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Loaded prompt config plus version metadata for span tracing."""

    config: dict[str, Any]
    version: str = "fallback"
    name: str = ""
    label: str = ""

    def as_metadata(self) -> dict[str, Any]:
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
      - Langfuse auth failure    → ERROR log, uses fallback
      - Prompt not found         → WARNING log, uses fallback
      - Langfuse unreachable     → WARNING log, uses fallback
      - Env vars not set         → uses fallback silently (expected in local dev)
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client: Any | None = client

    def _client_or_none(self) -> Any | None:
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
            log.error("PromptLoader: failed to initialise Langfuse client", exc_info=True)
        return self._client

    def get(
        self,
        name: str,
        fallback: dict[str, Any],
        label: str = "production",
        cache_ttl: int = 300,
    ) -> PromptResult:
        """Fetch prompt from Langfuse and merge with fallback.

        Args:
            name:      Prompt name in Langfuse (e.g. "researcher").
            fallback:  Dict of agent fields from YAML — used when Langfuse
                       is unavailable or the prompt does not exist yet.
            label:     Langfuse rollout label ("production", "staging", ...).
            cache_ttl: Seconds the SDK caches the prompt locally (default 300s).

        Returns:
            PromptResult with merged config and version metadata.
        """
        client = self._client_or_none()
        if client is None:
            return PromptResult(config=dict(fallback), version="fallback", name=name, label=label)

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
        except Exception as exc:
            exc_str = str(exc).lower()
            if "not found" in exc_str or "404" in exc_str:
                log.warning(
                    "Prompt '%s' (label=%s) not found in Langfuse — using YAML fallback",
                    name,
                    label,
                )
            elif "auth" in exc_str or "401" in exc_str or "403" in exc_str:
                log.error(
                    "Langfuse auth failure fetching prompt '%s' — check API keys",
                    name,
                    exc_info=True,
                )
            else:
                log.warning(
                    "Langfuse unreachable fetching prompt '%s' — using YAML fallback",
                    name,
                    exc_info=True,
                )

        return PromptResult(config=dict(fallback), version="fallback", name=name, label=label)
