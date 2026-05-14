"""HealthReportReaderTool — read a user-supplied health report file.

The file path is bound at tool-construction time (see tools/__init__.py's
TOOL_BUILDERS registry, which pulls it from inputs["health_report_path"]).
The agent invokes the tool with no arguments and receives the full report
text. When no file was uploaded, returns a short notice so the agent can
gracefully continue without the report.
"""
from __future__ import annotations

import logging
from pathlib import Path

from crewai.tools import BaseTool

log = logging.getLogger(__name__)

_NO_REPORT = "No health report was uploaded for this run."
_SUPPORTED = {".txt", ".md", ".pdf"}


class HealthReportReaderTool(BaseTool):
    name: str = "health_report_reader"
    description: str = (
        "Read the user's uploaded health report and return its full text. "
        "Call this once at the start of the analysis to learn about the user's "
        "medical conditions, medications, lab results, or other clinical context "
        "that should shape the fitness plan. Takes no arguments."
    )
    file_path: str = ""

    def _run(self, *args, **kwargs) -> str:
        # log via the logging module (not print) so the line bypasses
        # crews.common.kickoff_crew's stdout/stderr redirect and lands in
        # the platform log, where CrewAI's verbose prints do not.
        log.info("tool invoked: %s file_path=%r", self.name, self.file_path)
        if not self.file_path:
            return _NO_REPORT
        path = Path(self.file_path)
        if not path.is_file():
            return f"Health report file not found at {self.file_path!r}."
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED:
            return f"Unsupported health report format {suffix!r}; supported: {sorted(_SUPPORTED)}."
        if suffix == ".pdf":
            return _read_pdf(path)
        return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip() or "(PDF contained no extractable text.)"
