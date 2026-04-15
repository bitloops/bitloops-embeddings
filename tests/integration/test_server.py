from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bitloops_local_embeddings.server import create_app
from tests.support import FakeBackend


def test_health_endpoint_returns_runtime_status(tmp_path: Path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")
    client = TestClient(create_app(backend))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["model_id"] == "bge-m3"


def test_embed_endpoint_returns_embedding_payload(tmp_path: Path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")
    client = TestClient(create_app(backend))

    response = client.post("/embed", json={"texts": ["Hello World"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "bge-m3"
    assert payload["dimensions"] == 3
    assert payload["embeddings"][0]


def test_embed_endpoint_rejects_empty_batches(tmp_path: Path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")
    client = TestClient(create_app(backend))

    response = client.post("/embed", json={"texts": []})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_embed_endpoint_rejects_oversized_batches(tmp_path: Path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")
    client = TestClient(create_app(backend, max_batch_size=1))

    response = client.post("/embed", json={"texts": ["one", "two"]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_embed_endpoint_rejects_malformed_json(tmp_path: Path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")
    client = TestClient(create_app(backend))

    response = client.post(
        "/embed",
        content="{not-json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
