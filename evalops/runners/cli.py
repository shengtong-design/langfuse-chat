"""Headless EvalOps runner CLI.

Reproduces the original `scripts/run_experiment.py` behavior with explicit
flags. Phase 0 scope: dataset load + flow execution against Langfuse,
manifest persisted to `evalops/manifests/`. Scoring / report / gate land
in Phases 1 / 1 / 3 respectively.

Usage:
    python -m evalops.runners.cli \\
        --dataset crew-research-eval \\
        --crew researcher \\
        --prompt-label production
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
from evalops.crew_runner import make_task  # noqa: E402
from evalops.dataset_loader import load_dataset_items  # noqa: E402
from evalops.manifest import CrewRef, DatasetRef, ExperimentManifest  # noqa: E402


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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    experiment_name = args.experiment_name or (
        os.getenv("EXPERIMENT_PREFIX", "crewai-researcher-v1")
        + "-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    manifest = ExperimentManifest.start(experiment_name)
    manifest.crew = CrewRef(name=args.crew)
    manifest.environment = args.prompt_label

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
    manifest.finish()

    manifests_dir = _PROJECT_ROOT / "evalops" / "manifests"
    manifest_path = manifest.save(manifests_dir)

    print(
        f"\n[OK] Experiment '{experiment_name}' complete.\n"
        f"  Manifest: {manifest_path}\n"
        f"  Check Langfuse -> Datasets -> {args.dataset} -> Experiments"
    )


if __name__ == "__main__":
    main()
