from __future__ import annotations

import pytest

from bitloops_local_embeddings.errors import UnsupportedModelError
from bitloops_local_embeddings.registry import get_model_spec


def test_default_registry_contains_bge_m3() -> None:
    spec = get_model_spec("bge-m3")

    assert spec.model_id == "bge-m3"
    assert spec.upstream_model_id == "BAAI/bge-m3"
    assert spec.backend_name == "sentence-transformers"
    assert spec.dimensions == 1024


def test_unsupported_model_raises_clear_error() -> None:
    with pytest.raises(UnsupportedModelError, match="Unsupported model 'unknown-model'"):
        get_model_spec("unknown-model")
