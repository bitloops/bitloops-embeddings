from __future__ import annotations

import json
import logging
from pathlib import Path


LOGGER_NAME = "bitloops_embeddings"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        force=True,
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


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

