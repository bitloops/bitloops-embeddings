from __future__ import annotations

import json
from pathlib import Path

from bitloops_embeddings.cli import app
from typer.testing import CliRunner


def test_help_lists_public_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "embed" in result.stdout
    assert "serve" in result.stdout
    assert "describe" in result.stdout
    assert "daemon" in result.stdout


def test_embed_returns_json_to_stdout_and_output_file(
    fake_model,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    output_path = tmp_path / "embedding.json"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "embed",
            "--model",
            "bge-m3",
                "--input",
                "Hello World",
                "--cache-dir",
                str(cache_dir),
                "--output",
                str(output_path),
            ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    stdout_payload = json.loads(result.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_payload["model_id"] == "bge-m3"
    assert stdout_payload["dimensions"] == 3
    assert stdout_payload["embeddings"][0]
    assert stdout_payload["runtime"]["version"] == "0.1.1"
    assert file_payload == stdout_payload


def test_describe_returns_runtime_metadata(fake_model) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["describe", "--model", "bge-m3"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["model_id"] == "bge-m3"
    assert payload["backend_name"] == "fake-backend"
    assert payload["dimensions"] == 3
