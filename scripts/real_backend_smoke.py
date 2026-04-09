from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real backend smoke test.")
    parser.add_argument(
        "--binary",
        required=True,
        help="Executable to invoke. This may be a console script name or an absolute path.",
    )
    args = parser.parse_args()

    binary = args.binary
    port = reserve_free_port()

    run_embed_smoke(binary)
    run_server_smoke(binary, port)


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
