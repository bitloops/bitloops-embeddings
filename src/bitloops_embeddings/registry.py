from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bitloops_embeddings.backend.base import EmbeddingBackend
from bitloops_embeddings.backend.sentence_transformers_backend import (
    SentenceTransformersBackend,
)
from bitloops_embeddings.errors import UnsupportedModelError


BackendFactory = Callable[[Path, str], EmbeddingBackend]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    upstream_model_id: str
    backend_name: str
    dimensions: int
    factory: BackendFactory

    def create_backend(self, cache_dir: Path, requested_device: str = "auto") -> EmbeddingBackend:
        return self.factory(cache_dir, requested_device)


def _default_registry() -> dict[str, ModelSpec]:
    return {
        "bge-m3": ModelSpec(
            model_id="bge-m3",
            upstream_model_id="BAAI/bge-m3",
            backend_name="sentence-transformers",
            dimensions=1024,
            factory=lambda cache_dir, requested_device: SentenceTransformersBackend(
                model_id="bge-m3",
                upstream_model_id="BAAI/bge-m3",
                cache_dir=cache_dir,
                dimensions=1024,
                requested_device=requested_device,
            ),
        ),
    }


_MODEL_REGISTRY: dict[str, ModelSpec] = _default_registry()


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODEL_REGISTRY[model_id]
    except KeyError as exc:
        supported_models = ", ".join(sorted(_MODEL_REGISTRY))
        raise UnsupportedModelError(
            f"Unsupported model '{model_id}'. Supported models: {supported_models}."
        ) from exc


def register_model(spec: ModelSpec) -> None:
    _MODEL_REGISTRY[spec.model_id] = spec


def reset_model_registry() -> None:
    global _MODEL_REGISTRY
    _MODEL_REGISTRY = _default_registry()
