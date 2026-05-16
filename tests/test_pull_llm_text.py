"""Tests for crews.base._pull_llm_text — the LLM-text allowlist enforcement.

This invariant ensures only allowlisted fields (role/goal/backstory for agents,
description/expected_output for tasks) flow from Langfuse prompt config into
runtime Agent/Task construction. Wiring keys returned by Langfuse are silently
dropped so prompt edits can never override model selection, tool wiring, etc.
"""

from crews.base import (
    _AGENT_LLM_TEXT_FIELDS,
    _TASK_LLM_TEXT_FIELDS,
    _pull_llm_text,
)


def test_allowed_fields_flow_through_from_prompt_config():
    result = _pull_llm_text(
        prompt_config={"role": "R", "goal": "G", "backstory": "B"},
        fallback={},
        fields=_AGENT_LLM_TEXT_FIELDS,
    )
    assert result == {"role": "R", "goal": "G", "backstory": "B"}


def test_non_allowed_fields_dropped_even_if_langfuse_returns_them():
    result = _pull_llm_text(
        prompt_config={"role": "R", "model": "evil-override", "tools": ["evil_tool"]},
        fallback={},
        fields=_AGENT_LLM_TEXT_FIELDS,
    )
    assert result == {"role": "R"}
    assert "model" not in result
    assert "tools" not in result


def test_yaml_fallback_fills_fields_missing_from_prompt():
    result = _pull_llm_text(
        prompt_config={"role": "from_langfuse"},
        fallback={"goal": "from_yaml", "backstory": "from_yaml"},
        fields=_AGENT_LLM_TEXT_FIELDS,
    )
    assert result["role"] == "from_langfuse"
    assert result["goal"] == "from_yaml"
    assert result["backstory"] == "from_yaml"


def test_prompt_overrides_fallback_when_both_present():
    result = _pull_llm_text(
        prompt_config={"role": "from_langfuse"},
        fallback={"role": "from_yaml"},
        fields=_AGENT_LLM_TEXT_FIELDS,
    )
    assert result["role"] == "from_langfuse"


def test_missing_field_omitted_when_no_fallback():
    result = _pull_llm_text(
        prompt_config={"role": "R"},
        fallback={},
        fields=_AGENT_LLM_TEXT_FIELDS,
    )
    assert result == {"role": "R"}
    assert "goal" not in result


def test_task_fields_allowlist_applied():
    result = _pull_llm_text(
        prompt_config={
            "description": "D",
            "expected_output": "E",
            "model": "evil",
            "agent": "wrong_agent",
        },
        fallback={},
        fields=_TASK_LLM_TEXT_FIELDS,
    )
    assert result == {"description": "D", "expected_output": "E"}


def test_empty_prompt_config_and_empty_fallback_returns_empty_dict():
    result = _pull_llm_text(
        prompt_config={},
        fallback={},
        fields=_AGENT_LLM_TEXT_FIELDS,
    )
    assert result == {}
