# bitloops-embeddings

`bitloops-embeddings` is the Bitloops embeddings runtime repo. It ships two binaries from one source tree:

- `bitloops-local-embeddings`
  The heavyweight Python runtime for local model execution, HTTP serving, and stdio IPC.
- `bitloops-platform-embeddings`
  The lightweight Rust stdio daemon that forwards embedding requests to the public Bitloops platform gateway.

## Local Runtime

`bitloops-local-embeddings` keeps the existing local model flow:

- one-shot CLI embedding requests
- a long-lived local HTTP server
- a long-lived stdio daemon for Bitloops-managed IPC
- PyInstaller release bundles for supported desktop and server platforms

Install local development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Common commands:

```bash
bitloops-local-embeddings --help
bitloops-local-embeddings embed --model bge-m3 --input "Hello World"
bitloops-local-embeddings serve --model bge-m3
bitloops-local-embeddings daemon --model bge-m3
bitloops-local-embeddings describe --model bge-m3
```

The local cache resolution order is:

1. `--cache-dir`
2. `BITLOOPS_LOCAL_EMBEDDINGS_CACHE_DIR`
3. the platform cache directory for `bitloops-local-embeddings`

## Platform Runtime

`bitloops-platform-embeddings` is a Rust binary with the same daemon-facing contract as the local runtime, but it does not download or run models locally.

Run it manually:

```bash
BITLOOPS_PLATFORM_GATEWAY_TOKEN=secret \
bitloops-platform-embeddings \
  --gateway-url https://platform.example.com/v1/embeddings \
  --api-key-env BITLOOPS_PLATFORM_GATEWAY_TOKEN \
  daemon \
  --model bge-m3
```

Daemon contract:

- startup emits `{"event":"ready","protocol":1,"capabilities":["embed","ping","health","shutdown"]}`
- `ping` replies with `{"id":"...","ok":true,"pong":true}`
- `health` replies with `{"id":"...","ok":true,"status":"ok","model_loaded":true,"model":"..."}`
- `embed` replies with `{"id":"...","ok":true,"vectors":[...],"model":"..."}`
- `shutdown` replies with `{"id":"...","ok":true}`

Both binaries are validated against the shared protocol fixtures in [`tests/protocol_fixtures`](/Users/vasilis/Code/Bitloops/bitloops-embeddings/tests/protocol_fixtures).

## Packaging

Local runtime packaging:

```bash
python scripts/package_release.py --target x86_64-apple-darwin
```

Platform runtime packaging:

```bash
cargo build --release -p bitloops-platform-embeddings
python scripts/package_platform_release.py --target x86_64-apple-darwin
```

Release artefacts are published as:

- `bitloops-local-embeddings-v<version>-<target>`
- `bitloops-platform-embeddings-v<version>-<target>`

## Testing

Python local-runtime tests:

```bash
pytest
```

Rust platform-runtime tests:

```bash
cargo test -p bitloops-platform-embeddings
```
