"""Headless EvalOps runner CLI.

Thin wrapper around `evalops.runners.pipeline.run_pipeline`. See that
module for the actual pipeline implementation.

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
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.bootstrap import setup  # noqa: E402

setup()

from evalops.runners.pipeline import PipelineConfig, run_pipeline  # noqa: E402


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
        help="Comma-separated evaluator names to filter scores in the report.",
    )
    p.add_argument(
        "--wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait after experiment finishes before fetching scores. Default: 60.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else None
    )
    cfg = PipelineConfig(
        dataset=args.dataset,
        crew=args.crew,
        prompt_label=args.prompt_label,
        metrics=metrics,
        wait_seconds=args.wait_seconds,
        experiment_name=args.experiment_name,
    )
    result = run_pipeline(cfg)
    print(
        f"\n[OK] Experiment '{result.experiment_name}' complete.\n"
        f"  Items:    {result.item_count}\n"
        f"  Scores:   {result.score_count} ({result.distinct_evaluators} distinct evaluators)\n"
        f"  Manifest: {result.manifest_path}\n"
        f"  Report:   {result.report_path}\n"
        f"  Langfuse: Datasets -> {result.dataset} -> Experiments"
    )


if __name__ == "__main__":
    main()
