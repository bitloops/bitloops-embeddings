from __future__ import annotations

import json
from pathlib import Path

from bitloops_local_embeddings.cli import app
from bitloops_local_embeddings.version import __version__
from tests.support import FakeBackend, register_fake_model
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
    assert stdout_payload["runtime"]["version"] == __version__
    assert file_payload == stdout_payload


def test_embed_passes_requested_device_to_backend(tmp_path: Path) -> None:
    runner = CliRunner()
    captured_devices: list[str] = []

    register_fake_model(
        factory=lambda cache_dir, requested_device: (
            captured_devices.append(requested_device)
            or FakeBackend(
                cache_dir=cache_dir,
                requested_device=requested_device,
            )
        )
    )

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
            "--device",
            "cpu",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured_devices == ["cpu"]


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
