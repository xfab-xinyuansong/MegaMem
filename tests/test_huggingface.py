from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from megamem.huggingface import DATASET_ID, load_enterprise_rag


def test_load_enterprise_rag_uses_public_configs(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(dataset_id: str, config: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((dataset_id, config, kwargs))
        return {"config": config}

    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loaded = load_enterprise_rag(
        streaming=True,
        revision="snapshot-id",
        token="test-token",
    )

    assert loaded == {
        "documents": {"config": "documents"},
        "questions": {"config": "questions"},
    }
    assert calls == [
        (
            DATASET_ID,
            "documents",
            {
                "split": "test",
                "streaming": True,
                "revision": "snapshot-id",
                "token": "test-token",
            },
        ),
        (
            DATASET_ID,
            "questions",
            {
                "split": "test",
                "streaming": True,
                "revision": "snapshot-id",
                "token": "test-token",
            },
        ),
    ]
