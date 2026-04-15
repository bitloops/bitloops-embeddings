from __future__ import annotations

import io
import json
from pathlib import Path

from bitloops_local_embeddings.daemon import ProtocolWriter, handle_request, process_request_line, run_daemon
from tests.support import FakeBackend


FIXTURES = Path(__file__).resolve().parents[1] / "protocol_fixtures"


def load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FlushRecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_protocol_writer_appends_newline_and_flushes() -> None:
    stream = FlushRecordingStream()
    writer = ProtocolWriter(stream)

    writer.write_message({"event": "ready"})

    assert stream.getvalue() == "{\"event\": \"ready\"}\n"
    assert stream.flush_count == 1


def test_process_request_line_rejects_invalid_json(tmp_path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")

    response, should_shutdown = process_request_line("not-json", backend=backend, stderr=io.StringIO())

    assert should_shutdown is False
    assert response == {
        "ok": False,
        "error": {
            "code": "INVALID_JSON",
            "message": "could not parse request",
        },
    }


def test_handle_request_rejects_invalid_embed_texts(tmp_path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")

    response, should_shutdown = handle_request(
        {"id": "1", "cmd": "embed", "texts": []},
        backend=backend,
        stderr=io.StringIO(),
    )

    assert should_shutdown is False
    assert response["ok"] is False
    assert response["error"]["code"] == "BAD_REQUEST"


def test_handle_request_rejects_model_mismatch(tmp_path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")

    response, should_shutdown = handle_request(
        {"id": "2", "cmd": "ping", "model": "other-model"},
        backend=backend,
        stderr=io.StringIO(),
    )

    assert should_shutdown is False
    assert response["ok"] is False
    assert response["error"]["code"] == "BAD_REQUEST"


def test_handle_request_returns_unknown_command(tmp_path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")

    response, should_shutdown = handle_request(
        {"id": "3", "cmd": "frobnicate"},
        backend=backend,
        stderr=io.StringIO(),
    )

    assert should_shutdown is False
    assert response["ok"] is False
    assert response["error"]["code"] == "UNKNOWN_COMMAND"


def test_run_daemon_reuses_a_single_warm_backend(tmp_path) -> None:
    backend = FakeBackend(cache_dir=tmp_path / "cache")
    ping_request = load_fixture("ping_request.json")
    shutdown_request = load_fixture("shutdown_request.json")
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps(ping_request),
                json.dumps({"id": "2", "cmd": "embed", "texts": ["hello"]}),
                json.dumps({"id": "3", "cmd": "embed", "texts": ["world"]}),
                json.dumps(shutdown_request),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_daemon(backend, stdin=stdin, stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert backend.load_calls == 1
    assert backend.embed_calls == 2
    assert backend.close_calls == 1
    messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert messages[0] == load_fixture("ready_event.json")
    assert messages[1] == load_fixture("ping_response.json")
    assert messages[-1] == load_fixture("shutdown_response.json")
