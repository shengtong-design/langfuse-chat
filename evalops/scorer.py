"""LLM-as-a-Judge score collection and aggregation.

Fetches scores attached by Langfuse evaluation rules after a dataset
experiment run, then aggregates per evaluator.

Per the eval-gate spec Invariant 8: this module does NOT author scoring
logic. Score authoring lives in Langfuse LLM-as-a-Judge evaluators
configured in the Langfuse Cloud project. This module only collects.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import requests

DEFAULT_WAIT_SECONDS = 60
DEFAULT_PAGE_LIMIT = 100
TRACE_LABEL_MAX_LEN = 60
_TRACE_LABEL_KEYS = ("question", "query", "prompt", "message", "text", "input")


def fetch_trace_inputs(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch trace bodies (input/output/metadata) for the given trace IDs.

    Returns {trace_id: trace_body}. Failed fetches are skipped silently.
    """
    out: dict[str, dict[str, Any]] = {}
    for tid in {t for t in trace_ids if t}:
        try:
            resp = requests.get(
                f"{base_url.rstrip('/')}/api/public/traces/{tid}",
                auth=(public_key, secret_key),
                timeout=20,
            )
            if resp.ok:
                out[tid] = resp.json()
        except requests.RequestException:
            continue
    return out


def trace_label(trace_body: dict[str, Any], max_len: int = TRACE_LABEL_MAX_LEN) -> str:
    """Pull a short human-readable label out of a trace body's input."""
    inp = trace_body.get("input")
    candidate: str | None = None
    if isinstance(inp, dict):
        for key in _TRACE_LABEL_KEYS:
            v = inp.get(key)
            if isinstance(v, str) and v.strip():
                candidate = v
                break
        if candidate is None:
            candidate = str(inp)
    elif isinstance(inp, str):
        candidate = inp
    elif inp is not None:
        candidate = str(inp)
    if not candidate:
        return ""
    candidate = candidate.strip().replace("\r", " ").replace("\n", " ")
    if len(candidate) > max_len:
        candidate = candidate[: max_len - 1].rstrip() + "…"
    return candidate


def build_trace_labels(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    trace_ids: list[str],
) -> dict[str, str]:
    """Convenience: fetch trace inputs and return {trace_id: short_label}."""
    bodies = fetch_trace_inputs(
        base_url=base_url,
        public_key=public_key,
        secret_key=secret_key,
        trace_ids=trace_ids,
    )
    return {tid: trace_label(body) for tid, body in bodies.items()}


def fetch_scores(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    from_timestamp: str,
    evaluator_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch scores from Langfuse v2 scores API since `from_timestamp` (ISO-8601 UTC).

    If `evaluator_names` is provided, the result is filtered by score name.
    """
    scores: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = requests.get(
            f"{base_url.rstrip('/')}/api/public/v2/scores",
            auth=(public_key, secret_key),
            params={
                "fromTimestamp": from_timestamp,
                "limit": DEFAULT_PAGE_LIMIT,
                "page": page,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        scores.extend(payload.get("data", []))
        meta = payload.get("meta", {})
        if page >= meta.get("totalPages", 1) or not payload.get("data"):
            break
        page += 1

    if evaluator_names:
        wanted = set(evaluator_names)
        scores = [s for s in scores if s.get("name") in wanted]

    return scores


def aggregate(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-evaluator aggregate: count, mean, min, max of numeric values."""
    by_name: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        name = s.get("name")
        value = s.get("value")
        if name is None or value is None:
            continue
        try:
            by_name[name].append(float(value))
        except (TypeError, ValueError):
            continue
    return {
        name: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in by_name.items()
        if values
    }


def group_by_trace(scores: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Group scores by traceId; per-trace dict of {evaluator_name: value}."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for s in scores:
        trace_id = s.get("traceId")
        name = s.get("name")
        value = s.get("value")
        if not trace_id or not name or value is None:
            continue
        try:
            out[trace_id][name] = float(value)
        except (TypeError, ValueError):
            continue
    return dict(out)


def wait_then_fetch(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    from_timestamp: str,
    evaluator_names: list[str] | None = None,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
) -> list[dict[str, Any]]:
    """Block for `wait_seconds` so async judges can complete, then fetch.

    Phase 1: single sleep + fetch. Phase 1.5 enhancement: poll until score
    count stabilizes or max-wait is hit.
    """
    if wait_seconds > 0:
        print(f"[scorer] Waiting {wait_seconds}s for async judges to complete...")
        time.sleep(wait_seconds)
    return fetch_scores(
        base_url=base_url,
        public_key=public_key,
        secret_key=secret_key,
        from_timestamp=from_timestamp,
        evaluator_names=evaluator_names,
    )
