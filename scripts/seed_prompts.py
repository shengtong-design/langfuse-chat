"""
Seed Langfuse with initial prompt versions for all CrewAI concepts.

Walks agents/*.yaml and tasks/*.yaml, namespaces each prompt
(agent.<name>, task.<name>), and creates version 1 from the YAML fallback
labeled "production". Idempotent re-runs create new versions on existing
prompts; promote one by moving the label in the Langfuse UI.

Run from project root:
  py -3.12 scripts/seed_prompts.py

Env vars required:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
Optional:
  LANGFUSE_BASE_URL  (default: https://cloud.langfuse.com)
  SEED_LABEL         (default: production)
"""

import sys
import traceback
from pathlib import Path

# Put the project root on sys.path so `scripts.bootstrap` is importable when
# this file is run directly as `python scripts/seed_prompts.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.bootstrap import setup
setup()

import os
from typing import Iterable, Tuple

import yaml

from langfuse import Langfuse

from crews.base import (
    _AGENT_LLM_TEXT_FIELDS,
    _AGENT_PROMPT_NAMESPACE,
    _TASK_LLM_TEXT_FIELDS,
    _TASK_PROMPT_NAMESPACE,
    _namespaced,
)

_ROOT = Path(__file__).parent.parent
_AGENTS_DIR = _ROOT / "agents"
_TASKS_DIR = _ROOT / "tasks"

LABEL = os.getenv("SEED_LABEL", "production")

client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)


def _collect(
    directory: Path,
    namespace: str,
    key_field: str,
    llm_text_fields: tuple,
    prompt_body_field: str,
) -> Iterable[Tuple[str, dict, str]]:
    """Yield (langfuse_name, config_dict, prompt_body) for each concept YAML.

    config_dict is restricted to the LLM-text field set so seeding mirrors
    the runtime pull semantics (nothing else can flow into Langfuse via
    this script).
    """
    for yaml_file in sorted(directory.glob("*.yaml")):
        spec = yaml.safe_load(yaml_file.read_text()) or {}
        fallback = spec.get("fallback") or {}
        if not fallback:
            print(f"  ! skipping {yaml_file.name}: no 'fallback' block")
            continue
        bare_key = spec.get(key_field) or yaml_file.stem
        try:
            name = _namespaced(namespace, bare_key, source=f"{directory.name}/{yaml_file.name}")
        except ValueError as exc:
            print(f"  ! skipping {yaml_file.name}: {exc}")
            continue
        config = {f: fallback[f] for f in llm_text_fields if f in fallback}
        body = fallback.get(prompt_body_field, "")
        yield name, config, body


def seed() -> None:
    prompts = []
    prompts.extend(_collect(
        _AGENTS_DIR, _AGENT_PROMPT_NAMESPACE, "prompt_key",
        _AGENT_LLM_TEXT_FIELDS, prompt_body_field="backstory",
    ))
    prompts.extend(_collect(
        _TASKS_DIR, _TASK_PROMPT_NAMESPACE, "prompt_key",
        _TASK_LLM_TEXT_FIELDS, prompt_body_field="description",
    ))

    print(f"Seeding {len(prompts)} prompts (label='{LABEL}') ...\n")
    errors = 0
    for name, config, body in prompts:
        try:
            result = client.create_prompt(
                name=name,
                prompt=body,
                config=config,
                labels=[LABEL],
                type="text",
            )
            print(f"  [OK]   {name}  ->  version {result.version}")
        except Exception as e:
            print(f"  [FAIL] {name}  ->  {e}")
            traceback.print_exc()
            errors += 1

    client.flush()

    if errors:
        print(f"\n{errors} prompt(s) failed. Check credentials and retry.")
        sys.exit(1)

    print("\nDone. Open Langfuse -> Prompts to view and edit them.")
    print(
        "To promote a new version: edit in Langfuse UI, then move the "
        f"'{LABEL}' label to the new version."
    )


if __name__ == "__main__":
    seed()
