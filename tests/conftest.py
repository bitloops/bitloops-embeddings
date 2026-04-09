from __future__ import annotations

import pytest

from bitloops_embeddings.registry import reset_model_registry
from tests.support import (
    build_inference_failure_model,
    build_load_failure_model,
    register_fake_model,
)


@pytest.fixture(autouse=True)
def restore_model_registry() -> None:
    reset_model_registry()
    yield
    reset_model_registry()


@pytest.fixture
def fake_model():
    return register_fake_model()


@pytest.fixture
def load_failure_model():
    return build_load_failure_model()


@pytest.fixture
def inference_failure_model():
    return build_inference_failure_model()
