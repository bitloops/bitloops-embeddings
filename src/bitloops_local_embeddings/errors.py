from __future__ import annotations

from typing import Optional


class BitloopsEmbeddingsError(Exception):
    default_code = "runtime_error"
    default_status_code = 500

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.status_code = status_code or self.default_status_code


class UnsupportedModelError(BitloopsEmbeddingsError):
    default_code = "unsupported_model"
    default_status_code = 400


class UnsupportedDeviceError(BitloopsEmbeddingsError):
    default_code = "unsupported_device"
    default_status_code = 400


class BackendLoadError(BitloopsEmbeddingsError):
    default_code = "backend_load_error"
    default_status_code = 500


class InferenceError(BitloopsEmbeddingsError):
    default_code = "inference_error"
    default_status_code = 500


class InputValidationError(BitloopsEmbeddingsError):
    default_code = "invalid_request"
    default_status_code = 400
