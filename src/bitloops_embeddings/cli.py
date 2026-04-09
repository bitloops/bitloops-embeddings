from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from bitloops_embeddings.backend.base import EmbeddingBackend
from bitloops_embeddings.cache import ensure_cache_dir, resolve_cache_dir
from bitloops_embeddings.daemon import run_daemon
from bitloops_embeddings.errors import BitloopsEmbeddingsError
from bitloops_embeddings.logging_utils import configure_logging, log_event
from bitloops_embeddings.models import EmbeddingResponse, RuntimeInfo
from bitloops_embeddings.registry import get_model_spec
from bitloops_embeddings.server import (
    RUNTIME_NAME,
    build_describe_response,
    create_app,
    run_server,
)
from bitloops_embeddings.version import __version__


app = typer.Typer(
    add_completion=False,
    help="Managed local embeddings runtime for Bitloops.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    JSON = "json"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Transport(str, Enum):
    STDIO = "stdio"


def main() -> None:
    app(prog_name=RUNTIME_NAME)


@app.command()
def embed(
    model: Annotated[str, typer.Option("--model", help="Public model identifier.")],
    input_text: Annotated[str, typer.Option("--input", help="Input text to embed.")],
    format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format.", case_sensitive=False),
    ] = OutputFormat.JSON,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Override the model cache directory.",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            help="Optional path to mirror the JSON response.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
) -> None:
    configure_logging()
    if format is not OutputFormat.JSON:
        raise typer.BadParameter("Only JSON output is supported in v1.")

    try:
        backend = _build_backend(model=model, cache_dir=cache_dir)
        response = EmbeddingResponse(
            model_id=backend.model_id,
            dimensions=backend.dimensions,
            embeddings=backend.embed([input_text]),
            runtime=RuntimeInfo(name=RUNTIME_NAME, version=__version__),
        )
        _emit_json(response.model_dump_json(indent=2), output=output)
    except BitloopsEmbeddingsError as exc:
        _exit_with_error(exc)
    except Exception as exc:
        _exit_with_error(BitloopsEmbeddingsError(f"Unexpected runtime error: {exc}"))


@app.command()
def serve(
    model: Annotated[str, typer.Option("--model", help="Public model identifier.")],
    host: Annotated[str, typer.Option("--host", help="Host to bind the HTTP server to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind the HTTP server to.")] = 7719,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Override the model cache directory.",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="Server log verbosity.", case_sensitive=False),
    ] = LogLevel.INFO,
    log_file: Annotated[
        Optional[Path],
        typer.Option(
            "--log-file",
            help="Optional log file path. Defaults to the OS log sink for long-lived modes.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
    max_batch_size: Annotated[
        int,
        typer.Option("--max-batch-size", help="Maximum texts accepted by the /embed endpoint."),
    ] = 32,
) -> None:
    try:
        configure_logging(log_level.value, log_file=log_file, prefer_os_log=True)
        backend = _build_backend(model=model, cache_dir=cache_dir)
        backend.load()
        app_instance = create_app(backend, max_batch_size=max_batch_size)
        log_event(
            "server_start",
            model_id=backend.model_id,
            backend=backend.backend_name,
            host=host,
            port=port,
        )
        run_server(app_instance, host=host, port=port, log_level=log_level.value)
    except BitloopsEmbeddingsError as exc:
        _exit_with_error(exc)
    except Exception as exc:
        _exit_with_error(BitloopsEmbeddingsError(f"Unexpected runtime error: {exc}"))


@app.command()
def daemon(
    model: Annotated[str, typer.Option("--model", help="Public model identifier.")],
    transport: Annotated[
        Transport,
        typer.Option("--transport", help="IPC transport.", case_sensitive=False),
    ] = Transport.STDIO,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Override the model cache directory.",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="Daemon log verbosity.", case_sensitive=False),
    ] = LogLevel.INFO,
    log_file: Annotated[
        Optional[Path],
        typer.Option(
            "--log-file",
            help="Optional log file path. Defaults to the OS log sink for long-lived modes.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
) -> None:
    try:
        configure_logging(log_level.value, log_file=log_file, prefer_os_log=True)
        if transport is not Transport.STDIO:
            raise typer.BadParameter("Only stdio transport is supported in v1.")
        backend = _build_backend(model=model, cache_dir=cache_dir)
        raise typer.Exit(code=run_daemon(backend))
    except typer.BadParameter:
        raise
    except typer.Exit:
        raise
    except BitloopsEmbeddingsError as exc:
        _exit_with_error(exc)
    except Exception as exc:
        _exit_with_error(BitloopsEmbeddingsError(f"Unexpected runtime error: {exc}"))


@app.command()
def describe(
    model: Annotated[str, typer.Option("--model", help="Public model identifier.")],
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Override the model cache directory.",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = None,
) -> None:
    configure_logging()

    try:
        resolved_cache_dir = resolve_cache_dir(cache_dir)
        spec = get_model_spec(model)
        response = build_describe_response(
            model_id=spec.model_id,
            dimensions=spec.dimensions,
            backend_name=spec.backend_name,
            cache_dir=str(resolved_cache_dir),
        )
        typer.echo(response.model_dump_json(indent=2))
    except BitloopsEmbeddingsError as exc:
        _exit_with_error(exc)
    except Exception as exc:
        _exit_with_error(BitloopsEmbeddingsError(f"Unexpected runtime error: {exc}"))


def _build_backend(*, model: str, cache_dir: Optional[Path]) -> EmbeddingBackend:
    resolved_cache_dir = ensure_cache_dir(resolve_cache_dir(cache_dir))
    spec = get_model_spec(model)
    return spec.create_backend(resolved_cache_dir)


def _emit_json(payload: str, *, output: Optional[Path]) -> None:
    typer.echo(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{payload}\n", encoding="utf-8")


def _exit_with_error(exc: BitloopsEmbeddingsError) -> None:
    try:
        log_event("fatal_error", code=exc.code, message=str(exc))
    except Exception:
        pass
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)
