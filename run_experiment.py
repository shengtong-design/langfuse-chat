"""
Langfuse Dataset Experiment Runner — ResearchFlow.

Runs the research flow against the crew-research-eval dataset in Langfuse
and logs results for evaluation. Uses the same flows/crews/observability
architecture as crew_app.py.

Required env vars:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, OPENAI_API_KEY

Optional:
  LANGFUSE_BASE_URL  (default: https://cloud.langfuse.com)

Run:
  py -3.12 run_experiment.py
"""

import os

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

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

DATASET_NAME = "crew-research-eval"
EXPERIMENT_NAME = "crewai-researcher-v1-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> None:
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    print(f"Running experiment '{EXPERIMENT_NAME}' on {len(items)} items...")

    connectors = ConnectorManager([LangfuseConnector(langfuse)])

    def task(item):
        q = extract_question(item.input)
        print(f"Question: {q}")
        flow = ResearchFlow(connectors_factory=lambda: connectors, langfuse_client=langfuse)
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
        metadata={"framework": "crewai", "runner": "run_experiment.py"},
    )

    langfuse.flush()
    print(
        f"\n✅ Experiment '{EXPERIMENT_NAME}' complete! "
        f"Check Langfuse → Datasets → {DATASET_NAME} → Experiments"
    )


if __name__ == "__main__":
    main()
