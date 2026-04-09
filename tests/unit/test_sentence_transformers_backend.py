from __future__ import annotations

import sys
from types import ModuleType

from bitloops_embeddings.backend.sentence_transformers_backend import SentenceTransformersBackend


class FakeSentenceTransformer:
    attempts = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).attempts += 1
        if type(self).attempts < 3:
            raise RuntimeError("HTTP Error 503 thrown while requesting HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/config.json")

    def get_sentence_embedding_dimension(self) -> int:
        return 1024


def test_sentence_transformers_backend_retries_transient_load_failures(
    monkeypatch,
    tmp_path,
) -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.time.sleep", lambda _: None)
    FakeSentenceTransformer.attempts = 0

    backend = SentenceTransformersBackend(
        model_id="bge-m3",
        upstream_model_id="BAAI/bge-m3",
        cache_dir=tmp_path / "cache",
        dimensions=1024,
    )

    backend.load()

    assert backend.is_loaded is True
    assert backend.dimensions == 1024
    assert FakeSentenceTransformer.attempts == 3
