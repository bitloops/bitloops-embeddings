from __future__ import annotations

from pathlib import Path

import bitloops_local_embeddings.cli as cli_module
from bitloops_local_embeddings.cli import app
from typer.testing import CliRunner


def test_serve_writes_logs_to_the_requested_file(
    fake_model,
    monkeypatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    log_file = tmp_path / "logs" / "serve.log"

    def fake_run_server(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(cli_module, "run_server", fake_run_server)

    result = runner.invoke(
        app,
        [
            "serve",
            "--model",
            "bge-m3",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--log-file",
            str(log_file),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert log_file.exists()
    assert "event=\"server_start\"" in log_file.read_text(encoding="utf-8")
