"""Tools package — CrewAI BaseTool implementations + registry.

Agents reference tools by string key in their YAML (e.g.
``tools: [health_report_reader]``). The key is resolved here against
TOOL_BUILDERS: each builder receives the run's inputs dict and returns a
freshly-constructed BaseTool, so per-run state (uploaded file paths,
user-scoped credentials, ...) can be injected without leaking through
Langfuse-editable prompt text.

Add a new tool:
    1. Implement it as a BaseTool subclass in tools/<name>.py.
    2. Register a builder here under a unique snake_case key.
    3. Reference the key from the relevant agent's YAML under ``tools:``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from crewai.tools import BaseTool

from .health_report_reader_tool import HealthReportReaderTool

TOOL_BUILDERS: Dict[str, Callable[[Dict[str, Any]], BaseTool]] = {
    "health_report_reader": lambda inputs: HealthReportReaderTool(
        file_path=inputs.get("health_report_path", "") or "",
    ),
}

__all__ = ["TOOL_BUILDERS", "HealthReportReaderTool"]
