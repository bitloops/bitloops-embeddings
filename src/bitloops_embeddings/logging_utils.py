from __future__ import annotations

import json
import logging
import os
import platform
import sys
from logging.handlers import SysLogHandler
from pathlib import Path
from typing import Optional

from bitloops_embeddings.errors import BitloopsEmbeddingsError


LOGGER_NAME = "bitloops_embeddings"
LOG_FORMAT = "[bitloops-embeddings] level=%(levelname)s %(message)s"


def configure_logging(
    level: str = "INFO",
    *,
    log_file: Optional[Path] = None,
    prefer_os_log: bool = False,
) -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    handler = _build_handler(log_file=log_file, prefer_os_log=prefer_os_log)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(resolved_level)

    root_logger = logging.getLogger()
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)

    logging.captureWarnings(True)
    _normalise_library_loggers()
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").handlers.clear()
    logging.getLogger("uvicorn.error").propagate = True


def log_event(event: str, **fields: object) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    parts = [f"event={_format_value(event)}"]
    for key, value in fields.items():
        parts.append(f"{key}={_format_value(value)}")
    logger.info(" ".join(parts))


def _format_value(value: object) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return json.dumps(str(value), ensure_ascii=False)


def flush_logging() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def _build_handler(
    *,
    log_file: Optional[Path],
    prefer_os_log: bool,
) -> logging.Handler:
    if log_file is not None:
        return _build_file_handler(log_file)

    if prefer_os_log:
        os_log_handler = _build_os_log_handler()
        if os_log_handler is not None:
            return os_log_handler

    return logging.StreamHandler(sys.stderr)


def _build_file_handler(log_file: Path) -> logging.Handler:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return logging.FileHandler(log_file, encoding="utf-8")
    except OSError as exc:
        raise BitloopsEmbeddingsError(
            f"Failed to open log file '{log_file}'."
        ) from exc


def _build_os_log_handler() -> Optional[logging.Handler]:
    address = _resolve_syslog_address()
    if address is None:
        return None

    try:
        handler = SysLogHandler(address=address)
        handler.createSocket()
        return handler
    except OSError:
        return None


def _resolve_syslog_address() -> Optional[str]:
    system = platform.system()
    if system == "Darwin":
        address = "/var/run/syslog"
    elif system == "Linux":
        address = "/dev/log"
    else:
        return None

    if not os.path.exists(address):
        return None
    return address


def _normalise_library_loggers() -> None:
    library_loggers = (
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    )
    for logger_name in library_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
