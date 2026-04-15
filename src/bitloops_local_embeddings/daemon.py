from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from typing import Any, Optional, TextIO

from bitloops_local_embeddings.backend.base import EmbeddingBackend
from bitloops_local_embeddings.errors import BitloopsEmbeddingsError
from bitloops_local_embeddings.logging_utils import LOGGER_NAME, flush_logging, log_event


PROTOCOL_VERSION = 1
CAPABILITIES = ["embed", "ping", "health", "shutdown"]


class ProtocolWriter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write_message(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload, ensure_ascii=False))
        self._stream.write("\n")
        self._stream.flush()


def run_daemon(
    backend: EmbeddingBackend,
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    protocol_writer = ProtocolWriter(output_stream)
    startup_started_at = time.monotonic()
    cleaned_up = False
    reached_eof = False

    with _redirect_stdout_to(error_stream):
        backend.load()

    startup_duration_ms = int((time.monotonic() - startup_started_at) * 1000)
    log_event(
        "daemon_ready",
        model_id=backend.model_id,
        backend=backend.backend_name,
        duration_ms=startup_duration_ms,
    )
    protocol_writer.write_message(
        {
            "event": "ready",
            "protocol": PROTOCOL_VERSION,
            "capabilities": CAPABILITIES,
        }
    )

    try:
        for raw_line in input_stream:
            line = raw_line.strip()
            if not line:
                continue

            response, should_shutdown = process_request_line(
                line,
                backend=backend,
                stderr=error_stream,
            )
            protocol_writer.write_message(response)
            if should_shutdown:
                log_event("daemon_shutdown", model_id=backend.model_id)
                _cleanup(backend, stdout=output_stream, stderr=error_stream)
                cleaned_up = True
                return 0
        reached_eof = True
    finally:
        if not cleaned_up:
            if reached_eof:
                log_event("daemon_eof", model_id=backend.model_id)
            _cleanup(backend, stdout=output_stream, stderr=error_stream)

    return 0


def process_request_line(
    line: str,
    *,
    backend: EmbeddingBackend,
    stderr: Optional[TextIO] = None,
) -> tuple[dict[str, Any], bool]:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        log_event("daemon_invalid_json")
        return _error_response(
            code="INVALID_JSON",
            message="could not parse request",
        ), False

    return handle_request(
        request,
        backend=backend,
        stderr=stderr,
    )


def handle_request(
    request: Any,
    *,
    backend: EmbeddingBackend,
    stderr: Optional[TextIO] = None,
) -> tuple[dict[str, Any], bool]:
    error_stream = stderr or sys.stderr

    if not isinstance(request, dict):
        return _error_response(
            code="BAD_REQUEST",
            message="request must be a JSON object",
        ), False

    request_id = request.get("id")
    if not isinstance(request_id, str):
        return _error_response(
            code="BAD_REQUEST",
            message="id must be a string",
        ), False

    command = request.get("cmd")
    if not isinstance(command, str):
        return _error_response(
            request_id=request_id,
            code="BAD_REQUEST",
            message="cmd must be a string",
        ), False

    request_model = request.get("model")
    if request_model is not None and request_model != backend.model_id:
        return _error_response(
            request_id=request_id,
            code="BAD_REQUEST",
            message=f"model must match the daemon model: {backend.model_id}",
        ), False

    if command == "ping":
        log_event("daemon_request", request_id=request_id, cmd=command)
        return {"id": request_id, "ok": True, "pong": True}, False

    if command == "health":
        log_event("daemon_request", request_id=request_id, cmd=command)
        return {
            "id": request_id,
            "ok": True,
            "status": "ok",
            "model_loaded": backend.is_loaded,
            "model": backend.model_id,
        }, False

    if command == "embed":
        texts = request.get("texts")
        validation_error = _validate_texts(texts)
        if validation_error is not None:
            return _error_response(
                request_id=request_id,
                code="BAD_REQUEST",
                message=validation_error,
            ), False

        log_event(
            "daemon_request",
            request_id=request_id,
            cmd=command,
            texts=len(texts),
            model=backend.model_id,
        )
        try:
            with _redirect_stdout_to(error_stream):
                vectors = backend.embed(texts)
        except BitloopsEmbeddingsError as exc:
            logging.getLogger(LOGGER_NAME).exception(
                "daemon embed failed request_id=%s model=%s",
                request_id,
                backend.model_id,
            )
            return _error_response(
                request_id=request_id,
                code="INTERNAL",
                message="embedding inference failed",
            ), False
        except Exception:
            logging.getLogger(LOGGER_NAME).exception(
                "daemon embed failed request_id=%s model=%s",
                request_id,
                backend.model_id,
            )
            return _error_response(
                request_id=request_id,
                code="INTERNAL",
                message="embedding inference failed",
            ), False

        return {
            "id": request_id,
            "ok": True,
            "vectors": vectors,
            "model": backend.model_id,
        }, False

    if command == "shutdown":
        log_event("daemon_request", request_id=request_id, cmd=command)
        return {"id": request_id, "ok": True}, True

    return _error_response(
        request_id=request_id,
        code="UNKNOWN_COMMAND",
        message=f"unsupported cmd: {command}",
    ), False


def _validate_texts(texts: Any) -> Optional[str]:
    if not isinstance(texts, list) or not texts:
        return "texts must be a non-empty array of strings"

    if any(not isinstance(text, str) for text in texts):
        return "texts must be a non-empty array of strings"

    return None


def _error_response(
    *,
    code: str,
    message: str,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if request_id is not None:
        response["id"] = request_id
    return response


def _cleanup(
    backend: EmbeddingBackend,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    with _redirect_stdout_to(stderr):
        backend.close()
    flush_logging()
    stdout.flush()
    stderr.flush()


@contextlib.contextmanager
def _redirect_stdout_to(stream: TextIO):
    with contextlib.redirect_stdout(stream):
        yield
