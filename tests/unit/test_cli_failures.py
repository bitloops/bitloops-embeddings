from __future__ import annotations

from pathlib import Path

import bitloops_local_embeddings.cli as cli_module
from bitloops_local_embeddings.cli import app
from typer.testing import CliRunner


def test_embed_exits_non_zero_on_inference_failure(
    inference_failure_model,
    tmp_path: Path,
) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "embed",
            "--model",
            "bge-m3",
            "--input",
            "Hello World",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "Error: Embedding request failed." in result.stderr


def test_serve_exits_non_zero_on_model_load_failure(
    load_failure_model, monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    called = False

    def fake_run_server(*args, **kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli_module, "run_server", fake_run_server)

    result = runner.invoke(
        app,
        [
            "serve",
            "--model",
            "bge-m3",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert called is False
    assert "Error: Model load failed." in result.stderr
