"""EvalOps pipeline — runs an experiment end-to-end.

Shared entry point for `evalops.runners.cli` (headless) and
`evalops.runners.streamlit_tab` (UI). Produces a manifest JSON +
local Markdown report + Langfuse experiment run.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langfuse import Langfuse

from core.observability import ConnectorManager
from core.observability.langfuse_connector import LangfuseConnector
from evalops.crew_runner import get_flow_class, make_task
from evalops.dataset_loader import load_dataset_items
from evalops.manifest import CrewRef, DatasetRef, ExperimentManifest, FlowRef
from evalops.reporter import generate_report
from evalops.scorer import build_trace_labels, wait_then_fetch


@dataclass(frozen=True)
class PipelineConfig:
    dataset: str
    crew: str
    prompt_label: str
    metrics: list[str] | None = None
    wait_seconds: int = 60
    experiment_name: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    experiment_name: str
    item_count: int
    score_count: int
    distinct_evaluators: int
    manifest_path: Path
    report_path: Path
    langfuse_base_url: str
    dataset: str


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute one EvalOps experiment end-to-end."""
    experiment_name = config.experiment_name or (
        os.getenv("EXPERIMENT_PREFIX", "crewai-researcher-v1")
        + "-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    manifest = ExperimentManifest.start(experiment_name)
    manifest.crew = CrewRef(name=config.crew)
    manifest.environment = config.prompt_label

    flow_cls = get_flow_class(config.crew)
    manifest.flow = FlowRef(
        name=flow_cls.__name__,
        version=getattr(flow_cls, "flow_version", None),
    )
    manifest.metrics_requested = list(config.metrics or [])

    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=base_url,
    )

    items, dataset_meta = load_dataset_items(langfuse, config.dataset)
    manifest.dataset = DatasetRef(
        name=dataset_meta["name"],
        item_count=dataset_meta["item_count"],
    )

    print(f"Running experiment '{experiment_name}' on {len(items)} items...")

    connectors = ConnectorManager([LangfuseConnector(langfuse)])
    task = make_task(config.crew, connectors)

    langfuse.run_experiment(
        name=experiment_name,
        run_name=experiment_name,
        data=items,
        task=task,
        max_concurrency=1,
        metadata={
            "framework": "crewai",
            "runner": "evalops.runners.pipeline",
            "crew": config.crew,
            "prompt_label": config.prompt_label,
        },
    )
    langfuse.flush()

    scores = wait_then_fetch(
        base_url=base_url,
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        from_timestamp=manifest.started_at,
        evaluator_names=list(config.metrics) if config.metrics else None,
        wait_seconds=config.wait_seconds,
    )

    trace_ids = sorted({s.get("traceId") for s in scores if s.get("traceId")})
    trace_labels = build_trace_labels(
        base_url=base_url,
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        trace_ids=trace_ids,
    )

    manifest.finish()
    manifest_path = manifest.save(_PROJECT_ROOT / "evalops" / "manifests")
    report_path = generate_report(
        manifest,
        scores,
        _PROJECT_ROOT / "evalops" / "reports",
        trace_labels=trace_labels,
    )

    return PipelineResult(
        experiment_name=experiment_name,
        item_count=len(items),
        score_count=len(scores),
        distinct_evaluators=len({s.get("name") for s in scores if s.get("name")}),
        manifest_path=manifest_path,
        report_path=report_path,
        langfuse_base_url=base_url,
        dataset=config.dataset,
    )
