"""
Langfuse Dataset Experiment Runner — ResearchFlow.

Runs the research flow against a Langfuse dataset and logs results for
evaluation. Uses the same flows/crews/observability architecture as crew_app.py.

Required env vars:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, OPENAI_API_KEY

Optional:
  LANGFUSE_BASE_URL    (default: https://cloud.langfuse.com)
  DATASET_NAME         (default: crew-research-eval)
  EXPERIMENT_PREFIX    (default: crewai-researcher-v1)

Run from project root:
  py -3.12 scripts/run_experiment.py
"""

import sys
from pathlib import Path

# Put the project root on sys.path so `scripts.bootstrap` is importable when
# this file is run directly as `python scripts/run_experiment.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.bootstrap import setup

setup()

import os
from datetime import datetime

from langfuse import Langfuse

from core.observability import ConnectorManager
from core.observability.langfuse_connector import LangfuseConnector
from crews.common import extract_question
from flows.research_flow import ResearchFlow

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

DATASET_NAME = os.getenv("DATASET_NAME", "crew-research-eval")
EXPERIMENT_NAME = (
    os.getenv("EXPERIMENT_PREFIX", "crewai-researcher-v1")
    + "-"
    + datetime.now().strftime("%Y%m%d-%H%M%S")
)


def main() -> None:
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    print(f"Running experiment '{EXPERIMENT_NAME}' on {len(items)} items...")

    connectors = ConnectorManager([LangfuseConnector(langfuse)])

    def task(item):
        q = extract_question(item.input)
        print(f"Question: {q}")
        flow = ResearchFlow(connectors_factory=lambda: connectors)
        result = flow.kickoff(inputs={"question": q})
        answer = result.get("result", "") if isinstance(result, dict) else str(result)
        print(f"Answer (preview): {str(answer)[:100]}...")
        return answer

    langfuse.run_experiment(
        name=EXPERIMENT_NAME,
        run_name=EXPERIMENT_NAME,
        data=items,
        task=task,
        max_concurrency=1,
        metadata={"framework": "crewai", "runner": "scripts/run_experiment.py"},
    )

    langfuse.flush()
    print(
        f"\n✅ Experiment '{EXPERIMENT_NAME}' complete! "
        f"Check Langfuse → Datasets → {DATASET_NAME} → Experiments"
    )


if __name__ == "__main__":
    main()
