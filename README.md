# bitloops-embeddings

`bitloops-embeddings` is a managed local embeddings runtime for Bitloops. It provides:

- a one-shot CLI for simple embedding requests
- a long-lived local HTTP server for repeated requests
- a long-lived stdio daemon for process-managed IPC
- release packaging for major desktop and server operating systems

The first release is intentionally operational rather than retrieval-quality-complete. It focuses on a stable interface, model bootstrapping, hello-world inference, and releasable artefacts.

## Runtime model

The initial public model identifier is `bge-m3`.

- Public model id: `bge-m3`
- Upstream model id: `BAAI/bge-m3`
- Backend: `sentence-transformers`
- Device:
  - Apple Silicon macOS: `mps` when available, otherwise CPU
  - all other current targets: CPU
- Provisioning: first-run download into a local cache directory

The command and HTTP layers are written against an internal backend registry so additional models or inference backends can be added later without changing the user-facing contracts.

## Acceleration support

Current hardware acceleration support is intentionally limited in `v0.1.1`:

- `aarch64-apple-darwin`:
  - uses Apple Metal Performance Shaders (`mps`) automatically when available
  - falls back to CPU if MPS is unavailable
- `x86_64-apple-darwin`:
  - CPU only
- `x86_64-unknown-linux-gnu`:
  - CPU only
- `aarch64-unknown-linux-gnu`:
  - CPU only
- `x86_64-pc-windows-msvc`:
  - CPU only

The current release does not expose CUDA, ROCm, DirectML, or Intel GPU acceleration paths yet.

## Requirements

- Python `3.11` or `3.12`
- `pip`

## Local development

Create an environment and install the project with development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## CLI usage

Show the available commands:

```bash
bitloops-embeddings --help
```

Generate a single embedding:

```bash
bitloops-embeddings embed --model bge-m3 --input "Hello World"
```

Example response:

```json
{
  "model_id": "bge-m3",
  "dimensions": 1024,
  "embeddings": [[0.123, -0.456, 0.789]],
  "runtime": {
    "name": "bitloops-embeddings",
    "version": "0.1.1"
  }
}
```

Write the same JSON response to a file as well:

```bash
bitloops-embeddings embed \
  --model bge-m3 \
  --input "Hello World" \
  --output ./embedding.json
```

Inspect model metadata without loading the model:

```bash
bitloops-embeddings describe --model bge-m3
```

## Server usage

Start the local server:

```bash
bitloops-embeddings serve --model bge-m3
```

Defaults:

- host: `127.0.0.1`
- port: `7719`
- max batch size: `32`

Override the bind target:

```bash
bitloops-embeddings serve --model bge-m3 --host 127.0.0.1 --port 7719
```

Configure logging for long-lived modes:

```bash
bitloops-embeddings serve \
  --model bge-m3 \
  --log-level debug \
  --log-file ./bitloops-embeddings.log
```

### HTTP API

Health:

```bash
curl http://127.0.0.1:7719/health
```

Embed:

```bash
curl -X POST http://127.0.0.1:7719/embed \
  -H "content-type: application/json" \
  -d '{"texts":["Hello World"]}'
```

Response shape:

```json
{
  "model_id": "bge-m3",
  "dimensions": 1024,
  "embeddings": [[0.123, -0.456, 0.789]],
  "runtime": {
    "name": "bitloops-embeddings",
    "version": "0.1.1"
  }
}
```

Error shape:

```json
{
  "error": {
    "code": "runtime_error",
    "message": "..."
  }
}
```

## Daemon usage

Start the stdio daemon:

```bash
bitloops-embeddings daemon --model bge-m3
```

The daemon:

- loads the model once and keeps it warm
- reads newline-delimited JSON requests from `stdin`
- writes newline-delimited JSON protocol responses only to `stdout`
- writes logs and diagnostics to the configured log sink or, if needed, to `stderr`

Use a custom log file:

```bash
bitloops-embeddings daemon \
  --model bge-m3 \
  --log-level info \
  --log-file ./bitloops-embeddings-daemon.log
```

Ready event:

```json
{"event":"ready","protocol":1,"capabilities":["embed","ping","health","shutdown"]}
```

Example request:

```json
{"id":"1","cmd":"embed","texts":["hello","world"],"model":"bge-m3"}
```

Example response:

```json
{"id":"1","ok":true,"vectors":[[0.12,0.98],[-0.44,0.07]],"model":"bge-m3"}
```

Example error:

```json
{"id":"7","ok":false,"error":{"code":"UNKNOWN_COMMAND","message":"unsupported cmd: frobnicate"}}
```

The daemon exits cleanly on `shutdown` or when `stdin` reaches EOF.

## Cache directory resolution

Model cache resolution order:

1. `--cache-dir`
2. `BITLOOPS_EMBEDDINGS_CACHE_DIR`
3. platform default cache directory via `platformdirs`

Examples:

- macOS: `~/Library/Caches/bitloops-embeddings`
- Linux: `~/.cache/bitloops-embeddings`
- Windows: `%LOCALAPPDATA%/bitloops-embeddings/Cache`

## Packaging

Release packaging uses PyInstaller `--onedir` bundles. Each archive contains:

- the launchable runtime bundle
- `README.md`
- `LICENSE`

Create a local packaged artefact:

```bash
python scripts/package_release.py --target x86_64-apple-darwin
```

Run the real-model smoke test against an installed console script or packaged executable:

```bash
python scripts/real_backend_smoke.py --binary bitloops-embeddings
```

## GitHub Actions

The repository includes two workflows:

- `ci.yml`
  - installs dependencies
  - runs unit and integration tests
  - runs compile checks
  - validates the CLI help output
- `release.yml`
  - builds native bundles for the target matrix
  - packages archives
  - uploads artefacts
  - creates a GitHub Release for `v*.*.*` tags

## Troubleshooting

- The first `embed` or `serve` invocation downloads model files into the local cache. This can take a while on a cold machine.
- The first `daemon` invocation also downloads model files into the local cache if they are not already present.
- If model loading fails, check network access to Hugging Face and confirm the cache directory is writable.
- Long-lived modes support `--log-level` and `--log-file`. Without `--log-file`, `serve` and `daemon` use a best-effort OS log sink and fall back to `stderr` if the native sink is unavailable.
- The runtime does not log input texts by default.
