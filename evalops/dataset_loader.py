"""Langfuse dataset loader — typed iterable of items for the runner."""

from __future__ import annotations

from typing import Any

from langfuse import Langfuse


def load_dataset_items(client: Langfuse, dataset_name: str) -> tuple[list[Any], dict[str, Any]]:
    """Fetch a Langfuse dataset and return (items, dataset_meta)."""
    dataset = client.get_dataset(dataset_name)
    items = list(dataset.items)
    meta: dict[str, Any] = {"name": dataset_name, "item_count": len(items)}
    return items, meta
