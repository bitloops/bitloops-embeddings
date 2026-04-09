from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from bitloops_embeddings.backend.base import EmbeddingBackend
from bitloops_embeddings.errors import BitloopsEmbeddingsError, InputValidationError
from bitloops_embeddings.logging_utils import log_event
from bitloops_embeddings.models import (
    DescribeResponse,
    EmbedRequest,
    EmbeddingResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    RuntimeInfo,
)
from bitloops_embeddings.version import __version__


RUNTIME_NAME = "bitloops-embeddings"


def create_app(
    backend: EmbeddingBackend,
    *,
    max_batch_size: int = 32,
) -> FastAPI:
    app = FastAPI(title=RUNTIME_NAME, version=__version__)

    @app.exception_handler(BitloopsEmbeddingsError)
    async def handle_known_error(
        request: Request, exc: BitloopsEmbeddingsError
    ) -> JSONResponse:
        log_event(
            "request_failed",
            path=request.url.path,
            code=exc.code,
            message=str(exc),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_response(code=exc.code, message=str(exc)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        log_event(
            "request_failed",
            path=request.url.path,
            code="invalid_request",
            message=str(exc),
        )
        return JSONResponse(
            status_code=400,
            content=build_error_response(
                code="invalid_request",
                message="Malformed request payload.",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log_event(
            "request_failed",
            path=request.url.path,
            code="runtime_error",
            message=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content=build_error_response(
                code="runtime_error",
                message="An unexpected runtime error occurred.",
            ).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            model_id=backend.model_id,
            dimensions=backend.dimensions,
            runtime_version=__version__,
        )

    @app.post("/embed", response_model=EmbeddingResponse)
    async def embed(request: EmbedRequest) -> EmbeddingResponse:
        if not request.texts:
            raise InputValidationError("Request body must contain at least one text.")
        if len(request.texts) > max_batch_size:
            raise InputValidationError(
                f"Request body exceeds the max batch size of {max_batch_size}."
            )

        return EmbeddingResponse(
            model_id=backend.model_id,
            dimensions=backend.dimensions,
            embeddings=backend.embed(request.texts),
            runtime=RuntimeInfo(name=RUNTIME_NAME, version=__version__),
        )

    return app


def build_describe_response(
    *,
    model_id: str,
    dimensions: int,
    backend_name: str,
    cache_dir: str,
) -> DescribeResponse:
    return DescribeResponse(
        model_id=model_id,
        dimensions=dimensions,
        backend_name=backend_name,
        cache_dir=cache_dir,
        runtime_version=__version__,
    )


def build_error_response(*, code: str, message: str) -> ErrorResponse:
    return ErrorResponse(error=ErrorDetail(code=code, message=message))


def run_server(app: FastAPI, *, host: str, port: int, log_level: str) -> None:
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            access_log=False,
            log_config=None,
            use_colors=False,
        )
    )
    server.run()
    if not server.started:
        raise BitloopsEmbeddingsError(
            f"Failed to start the server on {host}:{port}.",
            code="startup_error",
            status_code=500,
        )
