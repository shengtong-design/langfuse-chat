"""Tests for crews.base._namespaced — the bare-key invariant at the Langfuse boundary.

This invariant ensures wiring (model, tools, retries) can never be overridden by a
Langfuse-side prompt edit, because the namespace prefix is added automatically and
dotted keys are rejected.
"""

import pytest

from crews.base import _AGENT_PROMPT_NAMESPACE, _TASK_PROMPT_NAMESPACE, _namespaced


def test_namespaced_happy_path_agent():
    assert (
        _namespaced(_AGENT_PROMPT_NAMESPACE, "researcher", source="agents/researcher.yaml")
        == "agent.researcher"
    )


def test_namespaced_happy_path_task():
    assert (
        _namespaced(_TASK_PROMPT_NAMESPACE, "research_task", source="tasks/research_task.yaml")
        == "task.research_task"
    )


def test_namespaced_rejects_dotted_key():
    with pytest.raises(ValueError, match="must be bare"):
        _namespaced(_AGENT_PROMPT_NAMESPACE, "agent.researcher", source="agents/researcher.yaml")


def test_namespaced_error_message_includes_source():
    with pytest.raises(ValueError, match="agents/researcher.yaml"):
        _namespaced(_AGENT_PROMPT_NAMESPACE, "a.b", source="agents/researcher.yaml")


def test_namespaced_error_message_includes_offending_key():
    with pytest.raises(ValueError, match="'a.b'"):
        _namespaced(_AGENT_PROMPT_NAMESPACE, "a.b", source="agents/x.yaml")
