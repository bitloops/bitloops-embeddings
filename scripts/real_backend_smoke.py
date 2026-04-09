from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request


SMOKE_RETRY_DELAYS_SECONDS = (10, 20, 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real backend smoke test.")
    parser.add_argument(
        "--binary",
        required=True,
        help="Executable to invoke. This may be a console script name or an absolute path.",
    )
    args = parser.parse_args()

    binary = args.binary
    run_with_retries("embed smoke", lambda: run_embed_smoke(binary))
    run_with_retries("server smoke", lambda: run_server_smoke(binary, reserve_free_port()))
    run_with_retries("daemon smoke", lambda: run_daemon_smoke(binary))


def run_with_retries(name: str, operation) -> None:
    total_attempts = len(SMOKE_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            operation()
            return
        except RuntimeError as exc:
            if attempt >= total_attempts:
                raise

            delay_seconds = SMOKE_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                f"{name} failed on attempt {attempt}/{total_attempts}: {exc}\nRetrying in {delay_seconds}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)


def run_embed_smoke(binary: str) -> None:
    completed = subprocess.run(
        [binary, "embed", "--model", "bge-m3", "--input", "Hello World"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Embed smoke failed with exit code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = json.loads(completed.stdout)
    if payload["model_id"] != "bge-m3":
        raise RuntimeError("Embed smoke returned an unexpected model id.")
    if not payload["embeddings"] or not payload["embeddings"][0]:
        raise RuntimeError("Embed smoke returned an empty embedding vector.")


def run_server_smoke(binary: str, port: int) -> None:
    with tempfile.TemporaryDirectory(prefix="bitloops-embeddings-serve-logs-") as temp_dir:
        log_file = Path(temp_dir) / "serve.log"
        process = subprocess.Popen(
            [
                binary,
                "serve",
                "--model",
                "bge-m3",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-file",
                str(log_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            wait_for_health(process, port)
            embed_response = http_post_json(
                f"http://127.0.0.1:{port}/embed",
                {"texts": ["Hello World"]},
            )
            if embed_response["model_id"] != "bge-m3":
                raise RuntimeError("Server smoke returned an unexpected model id.")
            if not embed_response["embeddings"] or not embed_response["embeddings"][0]:
                raise RuntimeError("Server smoke returned an empty embedding vector.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run_daemon_smoke(binary: str) -> None:
    with tempfile.TemporaryDirectory(prefix="bitloops-embeddings-daemon-logs-") as temp_dir:
        log_file = Path(temp_dir) / "daemon.log"
        process = subprocess.Popen(
            [
                binary,
                "daemon",
                "--model",
                "bge-m3",
                "--log-file",
                str(log_file),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            ready_event = read_json_line(process.stdout)
            if ready_event.get("event") != "ready":
                raise RuntimeError("Daemon smoke did not emit a ready event.")

            write_json_line(process.stdin, {"id": "1", "cmd": "ping"})
            ping_response = read_json_line(process.stdout)
            if ping_response != {"id": "1", "ok": True, "pong": True}:
                raise RuntimeError(f"Unexpected daemon ping response: {ping_response}")

            write_json_line(
                process.stdin,
                {"id": "2", "cmd": "embed", "texts": ["Hello World"]},
            )
            embed_response = read_json_line(process.stdout)
            if embed_response.get("model") != "bge-m3":
                raise RuntimeError("Daemon smoke returned an unexpected model id.")
            if not embed_response.get("vectors") or not embed_response["vectors"][0]:
                raise RuntimeError("Daemon smoke returned an empty embedding vector.")

            write_json_line(process.stdin, {"id": "3", "cmd": "shutdown"})
            shutdown_response = read_json_line(process.stdout)
            if shutdown_response != {"id": "3", "ok": True}:
                raise RuntimeError(f"Unexpected daemon shutdown response: {shutdown_response}")
            if process.wait(timeout=20) != 0:
                raise RuntimeError("Daemon smoke exited with a non-zero status.")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def write_json_line(stream, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def read_json_line(stream) -> dict[str, object]:
    line = stream.readline()
    if not line:
        raise RuntimeError("Expected a protocol message but reached EOF.")
    return json.loads(line)


def wait_for_health(process: subprocess.Popen[str], port: int, timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"

    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                "Server process exited before becoming healthy.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        try:
            payload = http_get_json(url)
            if payload.get("ok") is True:
                return
        except error.URLError:
            time.sleep(2)

    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
    raise RuntimeError(
        "Server did not become healthy within the timeout window.\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def http_get_json(url: str) -> dict[str, object]:
    with request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    main()
