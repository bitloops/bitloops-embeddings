from __future__ import annotations

from pathlib import Path
from typing import Any

from bitloops_embeddings.errors import BackendLoadError, InferenceError
from bitloops_embeddings.logging_utils import log_event


class SentenceTransformersBackend:
    def __init__(
        self,
        *,
        model_id: str,
        upstream_model_id: str,
        cache_dir: Path,
        dimensions: int,
    ) -> None:
        self._model_id = model_id
        self._upstream_model_id = upstream_model_id
        self._cache_dir = cache_dir
        self._dimensions = dimensions
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "sentence-transformers"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BackendLoadError(
                "The sentence-transformers dependency is not available. Install the project dependencies first."
            ) from exc

        log_event(
            "model_load_start",
            model_id=self.model_id,
            backend=self.backend_name,
            upstream_model_id=self._upstream_model_id,
            cache_dir=self._cache_dir,
        )
        try:
            self._model = SentenceTransformer(
                self._upstream_model_id,
                cache_folder=str(self._cache_dir),
                device="cpu",
            )
            detected_dimensions = self._model.get_sentence_embedding_dimension()
            if detected_dimensions is not None:
                self._dimensions = int(detected_dimensions)
        except Exception as exc:
            raise BackendLoadError(
                f"Failed to load model '{self.model_id}' from '{self._upstream_model_id}'."
            ) from exc

        log_event(
            "model_load_complete",
            model_id=self.model_id,
            backend=self.backend_name,
            dimensions=self.dimensions,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.load()
        try:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise InferenceError(
                f"Failed to generate embeddings for model '{self.model_id}'."
            ) from exc

        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [[float(value) for value in vector] for vector in vectors]
