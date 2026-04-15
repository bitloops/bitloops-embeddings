from __future__ import annotations

import logging
from pathlib import Path

from bitloops_local_embeddings.logging_utils import LOGGER_NAME, _build_os_log_handler, configure_logging, log_event


def test_configure_logging_writes_to_file_when_requested(tmp_path: Path) -> None:
    log_file = tmp_path / "runtime.log"

    configure_logging("INFO", log_file=log_file)
    log_event("configured", sink="file")

    payload = log_file.read_text(encoding="utf-8")
    assert "[bitloops-local-embeddings]" in payload
    assert "event=\"configured\"" in payload


def test_warning_level_filters_info_messages(tmp_path: Path) -> None:
    log_file = tmp_path / "warning.log"

    configure_logging("WARNING", log_file=log_file)
    log_event("info_should_be_filtered")
    logging.getLogger(LOGGER_NAME).warning("event=warning_message")

    payload = log_file.read_text(encoding="utf-8")
    assert "info_should_be_filtered" not in payload
    assert "warning_message" in payload


def test_os_log_handler_falls_back_when_socket_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("bitloops_local_embeddings.logging_utils._resolve_syslog_address", lambda: None)

    assert _build_os_log_handler() is None
