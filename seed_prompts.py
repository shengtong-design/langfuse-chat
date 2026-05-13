"""
Seed Langfuse with initial prompt versions for all crew agents.

Run once to bootstrap Langfuse prompt management. Each prompt is created
with the current YAML/hardcoded defaults as version 1, labeled "production".
After seeding, edit prompts in the Langfuse UI — no redeploy needed.

Run:
    py -3.12 seed_prompts.py

Env vars required:
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
Optional:
    LANGFUSE_BASE_URL  (default: https://cloud.langfuse.com)
    SEED_LABEL         (default: production)
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

LABEL = os.getenv("SEED_LABEL", "production")

# ---------------------------------------------------------------------------
# Prompt definitions — these mirror the current hardcoded / YAML defaults.
# Edit values here before seeding if you want a different starting point.
# ---------------------------------------------------------------------------

PROMPTS = [
    # ── Researcher crew ────────────────────────────────────────────────────
    {
        "name": "researcher_agent",
        "config": {
            "role": "Researcher",
            "goal": "Research the user's question and answer clearly and accurately.",
            "backstory": (
                "You are a diligent researcher who writes concise, "
                "well-structured answers with examples."
            ),
        },
    },

    # ── Fitness Training crew ──────────────────────────────────────────────
    {
        "name": "fitness_fitness_analyst",
        "config": {
            "role": "Fitness Analyst",
            "goal": (
                "Analyze user fitness goals, current fitness level, and preferences "
                "to create a comprehensive fitness profile"
            ),
            "backstory": (
                "You are an experienced fitness analyst with expertise in understanding "
                "different body types, fitness levels, and training goals. You have helped "
                "thousands of clients achieve their fitness objectives through careful "
                "analysis and personalized recommendations."
            ),
        },
    },
    {
        "name": "fitness_workout_designer",
        "config": {
            "role": "Workout Program Designer",
            "goal": (
                "Design effective and safe workout programs tailored to "
                "individual needs and goals"
            ),
            "backstory": (
                "You are a certified personal trainer and workout program designer with "
                "over 15 years of experience. You specialize in creating progressive "
                "training programs that maximize results while minimizing injury risk. "
                "You understand periodization, exercise selection, and program structure."
            ),
        },
    },
    {
        "name": "fitness_nutrition_advisor",
        "config": {
            "role": "Nutrition Advisor",
            "goal": "Provide nutrition guidance that supports the fitness goals and training program",
            "backstory": (
                "You are a certified nutritionist specializing in sports nutrition. "
                "You understand how to optimize nutrition for different training goals "
                "including muscle building, fat loss, and athletic performance."
            ),
        },
    },
]

# ---------------------------------------------------------------------------

def seed():
    print(f"Seeding {len(PROMPTS)} prompts (label='{LABEL}') ...\n")
    for p in PROMPTS:
        name = p["name"]
        try:
            result = client.create_prompt(
                name=name,
                prompt=p["config"].get("backstory", ""),  # text body (informational)
                config=p["config"],
                labels=[LABEL],
                type="text",
            )
            print(f"  ✓ {name}  →  version {result.version}")
        except Exception as e:
            print(f"  ✗ {name}  →  {e}")

    client.flush()
    print(f"\nDone. Open Langfuse → Prompts to view and edit them.")
    print(
        "To promote a new version: edit in Langfuse UI, then move the "
        f"'{LABEL}' label to the new version."
    )

if __name__ == "__main__":
    seed()
