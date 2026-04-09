from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path


def test_daemon_process_handles_requests_over_stdio(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "daemon.log"
    process = _start_daemon_process(tmp_path, log_file)

    try:
        ready_event = _read_json_line(process.stdout, timeout_seconds=10)
        assert ready_event == {
            "event": "ready",
            "protocol": 1,
            "capabilities": ["embed", "ping", "health", "shutdown"],
        }

        _send_request(process, {"id": "1", "cmd": "ping"})
        assert _read_json_line(process.stdout) == {"id": "1", "ok": True, "pong": True}

        _send_request(process, {"id": "2", "cmd": "health"})
        assert _read_json_line(process.stdout) == {
            "id": "2",
            "ok": True,
            "status": "ok",
            "model_loaded": True,
            "model": "bge-m3",
        }

        _send_request(process, {"id": "3", "cmd": "embed", "texts": ["hello", "world"]})
        embed_response = _read_json_line(process.stdout)
        assert embed_response["id"] == "3"
        assert embed_response["ok"] is True
        assert embed_response["model"] == "bge-m3"
        assert len(embed_response["vectors"]) == 2

        _send_request(process, {"id": "4", "cmd": "shutdown"})
        assert _read_json_line(process.stdout) == {"id": "4", "ok": True}
        assert process.wait(timeout=10) == 0

        assert process.stderr.read() == ""
        log_output = log_file.read_text(encoding="utf-8")
        assert "event=\"daemon_ready\"" in log_output
        assert "event=\"daemon_request\"" in log_output
    finally:
        _terminate_process(process)


def test_daemon_invalid_json_and_unknown_command_do_not_crash(tmp_path: Path) -> None:
    process = _start_daemon_process(tmp_path, tmp_path / "daemon.log")

    try:
        _read_json_line(process.stdout, timeout_seconds=10)

        process.stdin.write("not-json\n")
        process.stdin.flush()
        assert _read_json_line(process.stdout) == {
            "ok": False,
            "error": {
                "code": "INVALID_JSON",
                "message": "could not parse request",
            },
        }

        _send_request(process, {"id": "7", "cmd": "frobnicate"})
        assert _read_json_line(process.stdout) == {
            "id": "7",
            "ok": False,
            "error": {
                "code": "UNKNOWN_COMMAND",
                "message": "unsupported cmd: frobnicate",
            },
        }

        _send_request(process, {"id": "8", "cmd": "shutdown"})
        assert _read_json_line(process.stdout) == {"id": "8", "ok": True}
        assert process.wait(timeout=10) == 0
    finally:
        _terminate_process(process)


def test_daemon_exits_cleanly_on_stdin_eof(tmp_path: Path) -> None:
    process = _start_daemon_process(tmp_path, tmp_path / "daemon.log")

    try:
        _read_json_line(process.stdout, timeout_seconds=10)
        process.stdin.close()
        assert process.wait(timeout=10) == 0
    finally:
        _terminate_process(process)


def _start_daemon_process(tmp_path: Path, log_file: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.daemon_harness",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--log-file",
            str(log_file),
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _send_request(process: subprocess.Popen[str], payload: dict[str, object]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def _read_json_line(stream, timeout_seconds: int = 5) -> dict[str, object]:
    line = _read_line(stream, timeout_seconds=timeout_seconds)
    return json.loads(line)


def _read_line(stream, *, timeout_seconds: int = 5) -> str:
    result: dict[str, str] = {}

    def reader() -> None:
        result["line"] = stream.readline()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    assert not thread.is_alive(), "Timed out waiting for daemon output."
    return result["line"].strip()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
