"""EvalOps — Langfuse-driven evaluation runner around CrewAI flows.

See `architecture-standard-poc/drafts/eval-gate.md` in the cto-log for the
authoritative design spec. Invariants enforced here:

1. All EvalOps code lives in this top-level package; never interleaved
   with `scripts/`, `core/`, `flows/`, `crews/`, etc.
6. Promotion is a recommendation, never an applied action.
8. Metrics are Langfuse LLM-as-a-Judge evaluators, configured in
   Langfuse Cloud — never authored in code here.
"""
