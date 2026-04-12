from __future__ import annotations

import sys
from types import ModuleType

import pytest

from bitloops_embeddings.errors import UnsupportedDeviceError
from bitloops_embeddings.backend.sentence_transformers_backend import (
    SentenceTransformersBackend,
    _configure_tqdm_lock_for_single_process,
    resolve_inference_device,
)


class FakeSentenceTransformer:
    attempts = 0
    last_device: str | None = None

    def __init__(self, *args, **kwargs) -> None:
        type(self).attempts += 1
        type(self).last_device = kwargs.get("device")
        if type(self).attempts < 3:
            raise RuntimeError("HTTP Error 503 thrown while requesting HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/config.json")

    def get_sentence_embedding_dimension(self) -> int:
        return 1024


class FakeTqdm:
    lock = None

    @classmethod
    def set_lock(cls, lock) -> None:
        cls.lock = lock


def test_sentence_transformers_backend_retries_transient_load_failures(
    monkeypatch,
    tmp_path,
) -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.time.sleep", lambda _: None)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.system", lambda: "Linux")
    FakeSentenceTransformer.attempts = 0
    FakeSentenceTransformer.last_device = None

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
    assert FakeSentenceTransformer.last_device == "cpu"


def test_configure_tqdm_lock_uses_thread_lock(monkeypatch) -> None:
    fake_tqdm_package = ModuleType("tqdm")
    fake_tqdm_autonotebook = ModuleType("tqdm.autonotebook")
    fake_tqdm_autonotebook.tqdm = FakeTqdm
    FakeTqdm.lock = None

    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm_package)
    monkeypatch.setitem(sys.modules, "tqdm.autonotebook", fake_tqdm_autonotebook)
    monkeypatch.setattr(
        "bitloops_embeddings.backend.sentence_transformers_backend._TQDM_THREAD_LOCK_CONFIGURED",
        False,
    )

    _configure_tqdm_lock_for_single_process()

    assert FakeTqdm.lock is not None
    assert hasattr(FakeTqdm.lock, "acquire")
    assert hasattr(FakeTqdm.lock, "release")


def test_resolve_inference_device_prefers_mps_on_intel_mac_when_available(monkeypatch) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.backends = ModuleType("backends")
    fake_torch.backends.mps = type(
        "FakeMpsBackend",
        (),
        {
            "is_built": staticmethod(lambda: True),
            "is_available": staticmethod(lambda: True),
        },
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.machine", lambda: "x86_64")

    assert resolve_inference_device() == "mps"


def test_resolve_inference_device_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.backends = ModuleType("backends")
    fake_torch.backends.mps = type(
        "FakeMpsBackend",
        (),
        {
            "is_built": staticmethod(lambda: True),
            "is_available": staticmethod(lambda: False),
        },
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.machine", lambda: "x86_64")

    assert resolve_inference_device() == "cpu"


def test_resolve_inference_device_honours_cpu_override(monkeypatch) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.backends = ModuleType("backends")
    fake_torch.backends.mps = type(
        "FakeMpsBackend",
        (),
        {
            "is_built": staticmethod(lambda: True),
            "is_available": staticmethod(lambda: True),
        },
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.system", lambda: "Darwin")

    assert resolve_inference_device("cpu") == "cpu"


def test_resolve_inference_device_raises_clear_error_when_mps_is_requested_but_unavailable(
    monkeypatch,
) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.backends = ModuleType("backends")
    fake_torch.backends.mps = type(
        "FakeMpsBackend",
        (),
        {
            "is_built": staticmethod(lambda: True),
            "is_available": staticmethod(lambda: False),
        },
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("bitloops_embeddings.backend.sentence_transformers_backend.platform.system", lambda: "Darwin")

    with pytest.raises(UnsupportedDeviceError, match="MPS was requested but is unavailable"):
        resolve_inference_device("mps")
