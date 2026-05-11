"""
Langfuse Dataset Experiment Runner for CrewAI Researcher.

This script runs your CrewAI researcher agent against the crew-research-eval
dataset in Langfuse and logs results for evaluation.

Required env vars (same as crew_app.py):
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_BASE_URL
  - OPENAI_API_KEY

Run:
  py -3.12 run_experiment.py
"""

import os
import contextlib
import io
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from opentelemetry import trace
from crewai import Agent, Crew, Task

# --- Langfuse client ---
langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

DATASET_NAME = "crew-research-eval"
EXPERIMENT_NAME = "crewai-researcher-v1-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def run_crewai(question: str, lf) -> str:
    """Run the CrewAI researcher on a single question with nested Langfuse observations."""
    agent_spec = {
        "role": "Researcher",
        "goal": "Research the user's question and answer clearly and accurately.",
        "backstory": "You are a diligent researcher who writes concise, well-structured answers with examples.",
    }
    task_spec = {
        "description": f'Research the question: "{question}"',
        "expected_output": "A clear, concise answer with key points and 1-3 examples if applicable.",
    }

    researcher = Agent(
        role=agent_spec["role"],
        goal=agent_spec["goal"],
        backstory=agent_spec["backstory"],
        verbose=False,
        allow_delegation=False,
    )

    crew_task = Task(
        description=task_spec["description"],
        expected_output=task_spec["expected_output"],
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[crew_task], verbose=False)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    with lf.start_as_current_observation(
        name="crewai.research",
        as_type="chain",
        input={"question": question, "crew": {"agents": [agent_spec], "tasks": [task_spec]}},
        metadata={"framework": "crewai", "runner": "run_experiment.py"},
    ) as root:
        with root.start_as_current_observation(
            name="crew.kickoff",
            as_type="span",
            input={"question": question, "agents": [agent_spec], "tasks": [task_spec]},
        ) as kickoff:
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    result = crew.kickoff()
                output = str(result)
                kickoff.update(output={"result": output, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()})
                root.update(output={"result": output})
                return output
            except Exception as e:
                kickoff.update(output={"error": repr(e), "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}, level="ERROR")
                root.update(output={"error": repr(e)}, level="ERROR")
                raise


def main():
    # Load dataset from Langfuse
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    print(f"Running experiment '{EXPERIMENT_NAME}' on {len(items)} items...")

    def task(item):
        """
        Langfuse v4: use langfuse.run_experiment() to create traces for dataset runs.
        `item.input` may be a string or a JSON object depending on how the dataset was created.
        """
        q = item.input
        if isinstance(q, dict):
            q = q.get("question") or q.get("query") or q.get("input") or str(q)
        q = str(q)
        trace.get_current_span().update_name(q)
        print(f"Question: {q}")
        answer = run_crewai(q, langfuse)
        print(f"Answer (preview): {answer[:100]}...")
        return answer

    # This creates a dataset run + per-item traces automatically.
    # `run_name` shows up as the experiment run name in the Langfuse UI.
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
