"""Experiment manifest — captures exactly what was tested in an eval run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNNER_VERSION = "evalops-0.3.0"
SCHEMA_VERSION = "1.1"


@dataclass
class DatasetRef:
    name: str
    version: str | None = None
    item_count: int = 0


@dataclass
class CrewRef:
    name: str
    version: str | None = None


@dataclass
class FlowRef:
    name: str
    version: str | None = None


@dataclass
class PromptVersionRef:
    label: str
    version: str | None = None
    source: str = "langfuse"


@dataclass
class ExperimentManifest:
    experiment_name: str
    started_at: str
    schema_version: str = SCHEMA_VERSION
    completed_at: str | None = None
    dataset: DatasetRef | None = None
    crew: CrewRef | None = None
    flow: FlowRef | None = None
    agent_prompt_versions: dict[str, PromptVersionRef] = field(default_factory=dict)
    task_prompt_versions: dict[str, PromptVersionRef] = field(default_factory=dict)
    model: str | None = None
    environment: str | None = None
    deployment_sha: str | None = None
    runner_version: str = RUNNER_VERSION
    metrics_requested: list[str] = field(default_factory=list)
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def start(cls, experiment_name: str) -> "ExperimentManifest":
        return cls(
            experiment_name=experiment_name,
            started_at=_utcnow_iso(),
        )

    def finish(self) -> None:
        self.completed_at = _utcnow_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def save(self, dir_path: Path) -> Path:
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{self.experiment_name}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
