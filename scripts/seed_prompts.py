"""
Seed Langfuse with initial prompt versions for all crew agents.

Run once to bootstrap Langfuse prompt management. Each prompt is created
with the current YAML fallback defaults as version 1, labeled "production".
After seeding, edit prompts in the Langfuse UI — no redeploy needed.

Run from project root:
  py -3.12 scripts/seed_prompts.py

Env vars required:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
Optional:
  LANGFUSE_BASE_URL  (default: https://cloud.langfuse.com)
  SEED_LABEL         (default: production)
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

LABEL = os.getenv("SEED_LABEL", "production")
_AGENTS_DIR = Path(__file__).parent.parent / "agents"


def _load_prompts():
    """Build prompt list from agents/*.yaml files."""
    prompts = []
    for yaml_file in sorted(_AGENTS_DIR.glob("*.yaml")):
        spec = yaml.safe_load(yaml_file.read_text())
        fallback = spec.get("fallback", {})
        prompts.append({
            "name": spec["langfuse_prompt_key"],
            "config": fallback,
        })
    return prompts


def seed():
    prompts = _load_prompts()
    print(f"Seeding {len(prompts)} prompts (label='{LABEL}') ...\n")
    errors = 0
    for p in prompts:
        name = p["name"]
        try:
            result = client.create_prompt(
                name=name,
                prompt=p["config"].get("backstory", ""),
                config=p["config"],
                labels=[LABEL],
                type="text",
            )
            print(f"  ✓ {name}  →  version {result.version}")
        except Exception as e:
            print(f"  ✗ {name}  →  {e}")
            traceback.print_exc()
            errors += 1

    client.flush()

    if errors:
        print(f"\n⚠️  {errors} prompt(s) failed. Check credentials and retry.")
        sys.exit(1)

    print(f"\nDone. Open Langfuse → Prompts to view and edit them.")
    print(
        "To promote a new version: edit in Langfuse UI, then move the "
        f"'{LABEL}' label to the new version."
    )


if __name__ == "__main__":
    seed()
