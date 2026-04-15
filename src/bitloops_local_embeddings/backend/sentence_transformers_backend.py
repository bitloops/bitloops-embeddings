from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from threading import RLock
from typing import Any

from bitloops_local_embeddings.errors import (
    BackendLoadError,
    InferenceError,
    UnsupportedDeviceError,
)
from bitloops_local_embeddings.logging_utils import LOGGER_NAME, log_event


MODEL_LOAD_RETRY_DELAYS_SECONDS = (5, 10, 20)
_TQDM_THREAD_LOCK_CONFIGURED = False


class SentenceTransformersBackend:
    def __init__(
        self,
        *,
        model_id: str,
        upstream_model_id: str,
        cache_dir: Path,
        dimensions: int,
        requested_device: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._upstream_model_id = upstream_model_id
        self._cache_dir = cache_dir
        self._dimensions = dimensions
        self._model: Any = None
        self._requested_device = requested_device
        self._device = resolve_inference_device(requested_device=requested_device)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "sentence-transformers"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
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
        max_attempts = len(MODEL_LOAD_RETRY_DELAYS_SECONDS) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                self._model = SentenceTransformer(
                    self._upstream_model_id,
                    cache_folder=str(self._cache_dir),
                    device=self._device,
                )
                detected_dimensions = self._model.get_sentence_embedding_dimension()
                if detected_dimensions is not None:
                    self._dimensions = int(detected_dimensions)
                break
            except Exception as exc:
                self._model = None
                if attempt >= max_attempts or not _is_retryable_load_exception(exc):
                    logging.getLogger(LOGGER_NAME).exception(
                        "event=model_load_failed model_id=%s backend=%s upstream_model_id=%s device=%s",
                        self.model_id,
                        self.backend_name,
                        self._upstream_model_id,
                        self._device,
                    )
                    raise BackendLoadError(
                        f"Failed to load model '{self.model_id}' from '{self._upstream_model_id}'."
                    ) from exc

                delay_seconds = MODEL_LOAD_RETRY_DELAYS_SECONDS[attempt - 1]
                logging.getLogger(LOGGER_NAME).warning(
                    "event=model_load_retry model_id=%s backend=%s attempt=%s max_attempts=%s delay_seconds=%s reason=%s",
                    self.model_id,
                    self.backend_name,
                    attempt,
                    max_attempts,
                    delay_seconds,
                    str(exc),
                )
                time.sleep(delay_seconds)

        log_event(
            "model_load_complete",
            model_id=self.model_id,
            backend=self.backend_name,
            device=self._device,
            dimensions=self.dimensions,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.load()
        _configure_tqdm_lock_for_single_process()
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

    def close(self) -> None:
        self._model = None


def _is_retryable_load_exception(exc: Exception) -> bool:
    message = " ".join(_iter_exception_messages(exc)).lower()
    retryable_markers = (
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
        "connection error",
        "connection aborted",
        "connection reset",
        "read timed out",
        "timed out",
        "temporarily unavailable",
        "temporary failure",
        "service unavailable",
        "too many requests",
    )
    return any(marker in message for marker in retryable_markers)


def _iter_exception_messages(exc: Exception) -> list[str]:
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if message:
            messages.append(message)
        current = current.__cause__ or current.__context__

    return messages


def _configure_tqdm_lock_for_single_process() -> None:
    global _TQDM_THREAD_LOCK_CONFIGURED
    if _TQDM_THREAD_LOCK_CONFIGURED:
        return

    try:
        from tqdm.autonotebook import tqdm as auto_tqdm
    except ImportError:
        return

    # sentence-transformers always routes encode batches through tqdm, and
    # tqdm's default write lock uses multiprocessing.RLock on macOS/Linux.
    # That creates a semaphore even when progress bars are disabled, which can
    # surface noisy resource_tracker warnings at interpreter shutdown on
    # Python 3.12. A thread lock is sufficient for this single-process runtime.
    auto_tqdm.set_lock(RLock())
    _TQDM_THREAD_LOCK_CONFIGURED = True


def resolve_inference_device_for_request(requested_device: str) -> str:
    if requested_device == "auto":
        return "mps" if _is_mps_available() else "cpu"

    if requested_device == "cpu":
        return "cpu"

    if requested_device == "mps":
        unavailable_reason = _resolve_mps_unavailable_reason()
        if unavailable_reason is None:
            return "mps"
        raise UnsupportedDeviceError(
            f"MPS was requested but is unavailable: {unavailable_reason}"
        )

    raise UnsupportedDeviceError(
        f"Unsupported device '{requested_device}'. Supported devices: auto, cpu, mps."
    )


def resolve_inference_device(requested_device: str = "auto") -> str:
    return resolve_inference_device_for_request(requested_device)


def _is_mps_available() -> bool:
    return _resolve_mps_unavailable_reason() is None


def _resolve_mps_unavailable_reason() -> str | None:
    if platform.system() != "Darwin":
        return "MPS is only available on macOS."

    try:
        import torch
    except ImportError:
        return "PyTorch is not installed."

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is None:
        return "the installed PyTorch build does not expose torch.backends.mps."

    is_built = getattr(mps_backend, "is_built", None)
    if callable(is_built) and not is_built():
        return "the installed PyTorch build was not built with MPS support."

    is_available = getattr(mps_backend, "is_available", None)
    if callable(is_available) and is_available():
        return None

    return "macOS 12.3 or later and an MPS-enabled GPU are required."
