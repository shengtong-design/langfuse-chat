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
from evalops.flow_introspect import introspect_flow
from evalops.regression import (
    compare as compare_regression,
    find_baseline,
    find_baseline_in_langfuse,
)
from evalops.scorer import (
    aggregate,
    build_trace_bodies,
    trace_label,
    trace_output_text,
    wait_then_fetch,
)


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
    trace_bodies = build_trace_bodies(
        base_url=base_url,
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        trace_ids=trace_ids,
    )
    trace_labels = {tid: trace_label(body) for tid, body in trace_bodies.items()}
    trace_actuals = {tid: trace_output_text(body) for tid, body in trace_bodies.items()}

    expected_by_label: dict[str, str] = {}
    for item in items:
        label_key = trace_label({"input": item.input})
        exp = getattr(item, "expected_output", None) or getattr(item, "expectedOutput", None)
        if label_key and exp:
            expected_by_label[label_key] = str(exp)
    trace_expected = {
        tid: expected_by_label.get(label, "")
        for tid, label in trace_labels.items()
    }

    manifest.aggregates = aggregate(scores)
    manifest.finish()

    manifests_dir = _PROJECT_ROOT / "evalops" / "manifests"
    baseline = find_baseline(manifests_dir, manifest)
    if baseline is None:
        baseline = find_baseline_in_langfuse(
            base_url=base_url,
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            current=manifest,
        )
    regression = compare_regression(manifest, baseline) if baseline else None

    flow_graph = introspect_flow(flow_cls)

    manifest_path = manifest.save(manifests_dir)
    report_path = generate_report(
        manifest,
        scores,
        _PROJECT_ROOT / "evalops" / "reports",
        trace_labels=trace_labels,
        regression=regression,
        flow_graph=flow_graph,
        trace_expected=trace_expected,
        trace_actual=trace_actuals,
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
