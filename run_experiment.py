"""
Langfuse Dataset Experiment Runner for CrewAI Researcher.

Runs the researcher crew against the crew-research-eval dataset in Langfuse
and logs results for evaluation. Uses the same modular crew/connector
architecture as crew_app.py.

Required env vars:
  - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
  - OPENAI_API_KEY

Run:
  py -3.12 run_experiment.py
"""

import os

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from opentelemetry import trace
from langfuse import Langfuse

from crews.researcher import ResearcherCrew
from observability import ConnectorManager
from observability.langfuse_connector import LangfuseConnector

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

    obs = ConnectorManager([LangfuseConnector(langfuse)])
    crew = ResearcherCrew()

    def task(item):
        q = item.input
        if isinstance(q, dict):
            q = q.get("question") or q.get("query") or q.get("input") or str(q)
        q = str(q)
        trace.get_current_span().update_name(q)
        print(f"Question: {q}")
        result = crew.run({"question": q}, obs)
        answer = result["result"]
        print(f"Answer (preview): {answer[:100]}...")
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
