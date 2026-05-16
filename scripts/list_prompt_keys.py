"""List the Langfuse prompt keys this app would request at runtime.

Scans agents/*.yaml and tasks/*.yaml, derives each prompt_key the same way
crews.base does (falls back to filename stem), and prints the namespaced
keys (agent.<key> / task.<key>) the runtime would fetch from Langfuse.

Useful before a Langfuse-side prompt audit, label-lifecycle changes, or any
work that needs the canonical list of keys without booting the full app.

Exit codes:
  0  success
  1  YAML key validation failed (bare-key invariant breach in crews.base)
  2  unexpected error (missing dir, malformed YAML)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put the project root on sys.path so `scripts.bootstrap` is importable when
# this file is run directly as `py -3.12 scripts/list_prompt_keys.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.bootstrap import setup  # noqa: E402

setup()

import yaml  # noqa: E402

from crews.base import (  # noqa: E402
    _AGENT_PROMPT_NAMESPACE,
    _TASK_PROMPT_NAMESPACE,
    _namespaced,
)

ROOT = Path(__file__).parent.parent
AGENTS_DIR = ROOT / "agents"
TASKS_DIR = ROOT / "tasks"


def _collect(yaml_dir: Path, namespace: str, source_prefix: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(yaml_dir.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text()) or {}
        stem = path.stem
        prompt_key = spec.get("prompt_key") or stem
        source = f"{source_prefix}/{path.name}"
        langfuse_key = _namespaced(namespace, prompt_key, source=source)
        rows.append((langfuse_key, source))
    return rows


def main() -> int:
    if not AGENTS_DIR.is_dir():
        print(f"[ERROR] agents directory not found: {AGENTS_DIR}", file=sys.stderr)
        return 2
    if not TASKS_DIR.is_dir():
        print(f"[ERROR] tasks directory not found: {TASKS_DIR}", file=sys.stderr)
        return 2

    try:
        agent_rows = _collect(AGENTS_DIR, _AGENT_PROMPT_NAMESPACE, "agents")
        task_rows = _collect(TASKS_DIR, _TASK_PROMPT_NAMESPACE, "tasks")
    except ValueError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    width = max(
        (len(key) for key, _ in agent_rows + task_rows),
        default=0,
    )

    print("[agents]")
    for key, source in agent_rows:
        print(f"  {key.ljust(width)}  <-  {source}")
    print()
    print("[tasks]")
    for key, source in task_rows:
        print(f"  {key.ljust(width)}  <-  {source}")
    print()
    print(f"[OK] {len(agent_rows)} agents, {len(task_rows)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
