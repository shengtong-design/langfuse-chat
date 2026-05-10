"""
Run a simple CrewAI agent and trace execution to Langfuse using the Langfuse Python SDK
(no OpenTelemetry).

Required env vars:
  - LANGFUSE_PUBLIC_KEY=...
  - LANGFUSE_SECRET_KEY=...
  - LANGFUSE_BASE_URL=https://cloud.langfuse.com   (or your self-hosted URL)

You also need an LLM provider configured for CrewAI, e.g.:
  - OPENAI_API_KEY=...

Install deps (typical):
  - pip install crewai langfuse
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import sys
from typing import Any


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_langfuse():
    try:
        from langfuse import Langfuse
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: langfuse. Install it with:\n"
            "  pip install langfuse\n"
            "Then re-run this script."
        ) from e

    public_key = _require_env("LANGFUSE_PUBLIC_KEY")
    secret_key = _require_env("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)


def run_crewai(langfuse: Any) -> str:
    try:
        from crewai import Agent, Crew, Task
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: crewai. Install it with:\n"
            "  pip install crewai\n"
            "Then re-run this script."
        ) from e

    prompt = (
        'Research the question: "What is AI?" '
        "Provide a clear definition, key subfields, and 2 practical examples."
    )

    agent_spec = {
        "role": "Researcher",
        "goal": "Research the topic and explain it clearly and accurately.",
        "backstory": "You are a diligent researcher who writes concise, well-structured answers.",
        "verbose": True,
        "allow_delegation": False,
    }

    researcher = Agent(
        role=agent_spec["role"],
        goal=agent_spec["goal"],
        backstory=agent_spec["backstory"],
        verbose=agent_spec["verbose"],
        allow_delegation=agent_spec["allow_delegation"],
    )

    task_spec = {
        "description": prompt,
        "expected_output": "A concise explanation of AI with bullet points and examples.",
    }
    task = Task(
        description=task_spec["description"],
        expected_output=task_spec["expected_output"],
        agent=researcher,
    )

    crew_spec = {
        "verbose": True,
        "agents": [agent_spec],
        "tasks": [task_spec],
    }
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=crew_spec["verbose"],
    )

    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # Root observation for the full CrewAI run. We attach everything we know as input/metadata.
    with langfuse.start_as_current_observation(
        name="crewai.run",
        as_type="chain",
        input={
            "crew": crew_spec,
            "runtime": runtime,
        },
        metadata={
            "framework": "crewai",
            "topic": "What is AI?",
        },
    ) as root:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Kickoff span, plus capture all console output CrewAI emits.
        with root.start_as_current_observation(
            name="crew.kickoff",
            as_type="span",
            input={
                "task_descriptions": [task_spec["description"]],
                "expected_outputs": [task_spec["expected_output"]],
                "agents": [agent_spec],
            },
            metadata={"crew_verbose": crew_spec["verbose"]},
        ) as kickoff:
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    result = crew.kickoff()

                output = str(result)
                kickoff.update(
                    output={
                        "result": output,
                        "stdout": stdout_buf.getvalue(),
                        "stderr": stderr_buf.getvalue(),
                    }
                )
                root.update(
                    output={
                        "result": output,
                    }
                )
                return output
            except Exception as e:
                kickoff.update(
                    output={
                        "error": repr(e),
                        "stdout": stdout_buf.getvalue(),
                        "stderr": stderr_buf.getvalue(),
                    },
                    level="ERROR",
                )
                root.update(output={"error": repr(e)}, level="ERROR")
                raise
            finally:
                try:
                    langfuse.flush()
                except Exception:
                    pass


def main() -> int:
    langfuse = get_langfuse()
    output = run_crewai(langfuse)
    print("\n=== CrewAI result ===\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

