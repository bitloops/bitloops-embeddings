from __future__ import annotations

import json

from bitloops_embeddings.models import EmbeddingResponse, ErrorDetail, ErrorResponse, RuntimeInfo


def test_embedding_response_serialises_runtime_metadata() -> None:
    response = EmbeddingResponse(
        model_id="bge-m3",
        dimensions=3,
        embeddings=[[0.1, -0.2, 0.3]],
        runtime=RuntimeInfo(name="bitloops-embeddings", version="0.1.1"),
    )

    payload = json.loads(response.model_dump_json())

    assert payload["model_id"] == "bge-m3"
    assert payload["runtime"]["name"] == "bitloops-embeddings"
    assert payload["embeddings"][0]


def test_error_response_serialises_consistently() -> None:
    response = ErrorResponse(
        error=ErrorDetail(code="runtime_error", message="Something went wrong.")
    )

    payload = json.loads(response.model_dump_json())

    assert payload == {
        "error": {
            "code": "runtime_error",
            "message": "Something went wrong.",
        }
    }
