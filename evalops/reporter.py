"""Eval report generator — Markdown only, local-only.

Phase 0 stub. Phase 1 implementation renders sections 1-9 of the report
contract defined in the eval-gate spec. The output file is written to
`evalops/reports/{experiment_name}.md` and is git-ignored.
"""

from __future__ import annotations

from pathlib import Path

from evalops.manifest import ExperimentManifest


def generate_report(manifest: ExperimentManifest, reports_dir: Path) -> Path:
    """Render a Markdown report for the run; return the written file path."""
    raise NotImplementedError("Phase 1: implement Markdown report rendering.")
