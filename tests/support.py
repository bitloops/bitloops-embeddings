from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from bitloops_embeddings.errors import BackendLoadError, InferenceError
from bitloops_embeddings.registry import ModelSpec, register_model


class FakeBackend:
    def __init__(
        self,
        *,
        cache_dir: Path,
        requested_device: str = "auto",
        model_id: str = "bge-m3",
        dimensions: int = 3,
        vector: Optional[list[float]] = None,
        load_error: Optional[Exception] = None,
        embed_error: Optional[Exception] = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._requested_device = requested_device
        self._model_id = model_id
        self._dimensions = dimensions
        self._vector = vector or [0.1, -0.2, 0.3]
        self._load_error = load_error
        self._embed_error = embed_error
        self._loaded = False
        self.load_calls = 0
        self.embed_calls = 0
        self.close_calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "fake-backend"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        self.load_calls += 1
        if self._load_error is not None:
            raise self._load_error
        self._loaded = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.load()
        if self._embed_error is not None:
            raise self._embed_error
        return [list(self._vector) for _ in texts]

    def close(self) -> None:
        self.close_calls += 1
        self._loaded = False


def register_fake_model(
    factory: Optional[Callable[[Path, str], FakeBackend]] = None,
    *,
    dimensions: int = 3,
) -> ModelSpec:
    spec = ModelSpec(
        model_id="bge-m3",
        upstream_model_id="test/bge-m3",
        backend_name="fake-backend",
        dimensions=dimensions,
        factory=factory
        or (
            lambda cache_dir, requested_device: FakeBackend(
                cache_dir=cache_dir,
                requested_device=requested_device,
                dimensions=dimensions,
            )
        ),
    )
    register_model(spec)
    return spec


def build_load_failure_model() -> ModelSpec:
    return register_fake_model(
        factory=lambda cache_dir, requested_device: FakeBackend(
            cache_dir=cache_dir,
            requested_device=requested_device,
            load_error=BackendLoadError("Model load failed."),
        )
    )


def build_inference_failure_model() -> ModelSpec:
    return register_fake_model(
        factory=lambda cache_dir, requested_device: FakeBackend(
            cache_dir=cache_dir,
            requested_device=requested_device,
            embed_error=InferenceError("Embedding request failed."),
        )
    )
