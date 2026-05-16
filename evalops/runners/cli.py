"""Headless EvalOps runner CLI.

Phase 1 vertical slice: dataset experiment + Langfuse LLM-as-a-Judge
score collection + Markdown report writing.

Usage:
    python -m evalops.runners.cli \\
        --dataset crew-research-eval \\
        --crew researcher \\
        --prompt-label production \\
        [--metrics Conciseness,Hallucination,Correctness] \\
        [--wait-seconds 60]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Project root on sys.path so `core`, `crews`, `flows`, `scripts` import
# when this file is run as `python -m evalops.runners.cli`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.bootstrap import setup  # noqa: E402

setup()

from langfuse import Langfuse  # noqa: E402

from core.observability import ConnectorManager  # noqa: E402
from core.observability.langfuse_connector import LangfuseConnector  # noqa: E402
from evalops.crew_runner import get_flow_class, make_task  # noqa: E402
from evalops.dataset_loader import load_dataset_items  # noqa: E402
from evalops.manifest import CrewRef, DatasetRef, ExperimentManifest, FlowRef  # noqa: E402
from evalops.reporter import generate_report  # noqa: E402
from evalops.scorer import wait_then_fetch  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="evalops.runners.cli")
    p.add_argument("--dataset", default=os.getenv("DATASET_NAME", "crew-research-eval"))
    p.add_argument("--crew", default="researcher")
    p.add_argument("--prompt-label", default=os.getenv("PROMPT_LABEL", "production"))
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Override experiment name. Defaults to {EXPERIMENT_PREFIX}-{timestamp}.",
    )
    p.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated evaluator names to include in the report. "
             "Defaults to all scores collected in the run window.",
    )
    p.add_argument(
        "--wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait after experiment finishes before fetching scores. "
             "Allows async LLM-as-a-Judge evaluators to complete. Default: 60.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    experiment_name = args.experiment_name or (
        os.getenv("EXPERIMENT_PREFIX", "crewai-researcher-v1")
        + "-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    evaluator_names = (
        [n.strip() for n in args.metrics.split(",") if n.strip()]
        if args.metrics
        else None
    )

    manifest = ExperimentManifest.start(experiment_name)
    manifest.crew = CrewRef(name=args.crew)
    manifest.environment = args.prompt_label

    flow_cls = get_flow_class(args.crew)
    manifest.flow = FlowRef(name=flow_cls.__name__, version=getattr(flow_cls, "flow_version", None))
    manifest.metrics_requested = evaluator_names or []

    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )

    items, dataset_meta = load_dataset_items(langfuse, args.dataset)
    manifest.dataset = DatasetRef(
        name=dataset_meta["name"], item_count=dataset_meta["item_count"]
    )

    print(f"Running experiment '{experiment_name}' on {len(items)} items...")

    connectors = ConnectorManager([LangfuseConnector(langfuse)])
    task = make_task(args.crew, connectors)

    langfuse.run_experiment(
        name=experiment_name,
        run_name=experiment_name,
        data=items,
        task=task,
        max_concurrency=1,
        metadata={
            "framework": "crewai",
            "runner": "evalops.runners.cli",
            "crew": args.crew,
            "prompt_label": args.prompt_label,
        },
    )

    langfuse.flush()

    scores = wait_then_fetch(
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        from_timestamp=manifest.started_at,
        evaluator_names=evaluator_names,
        wait_seconds=args.wait_seconds,
    )

    manifest.finish()

    manifests_dir = _PROJECT_ROOT / "evalops" / "manifests"
    manifest_path = manifest.save(manifests_dir)

    reports_dir = _PROJECT_ROOT / "evalops" / "reports"
    report_path = generate_report(manifest, scores, reports_dir)

    print(
        f"\n[OK] Experiment '{experiment_name}' complete.\n"
        f"  Items:    {len(items)}\n"
        f"  Scores:   {len(scores)} fetched ({len({s.get('name') for s in scores}) - (1 if any(s.get('name') is None for s in scores) else 0)} distinct evaluators)\n"
        f"  Manifest: {manifest_path}\n"
        f"  Report:   {report_path}\n"
        f"  Langfuse: Datasets -> {args.dataset} -> Experiments"
    )


if __name__ == "__main__":
    main()
