use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

const READY_PROTOCOL_VERSION: u32 = 1;
const CAPABILITIES: &[&str] = &["embed", "ping", "health", "shutdown"];
const DEFAULT_PLATFORM_EMBEDDINGS_GATEWAY_URL: &str = "https://platform.bitloops.net/v1/embeddings";
const PLATFORM_GATEWAY_URL_ENV: &str = "BITLOOPS_PLATFORM_GATEWAY_URL";
const DEFAULT_PLATFORM_API_KEY_ENV: &str = "BITLOOPS_PLATFORM_GATEWAY_TOKEN";

#[derive(Debug, Parser)]
#[command(name = "bitloops-platform-embeddings")]
pub struct Cli {
    #[arg(long)]
    pub gateway_url: Option<String>,
    #[arg(long, default_value = DEFAULT_PLATFORM_API_KEY_ENV)]
    pub api_key_env: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Daemon(DaemonArgs),
}

#[derive(Debug, Clone, clap::Args)]
pub struct DaemonArgs {
    #[arg(long)]
    pub model: String,
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

pub fn run(cli: Cli) -> Result<()> {
    init_tracing();

    match cli.command {
        Commands::Daemon(args) => {
            let gateway_url = resolve_gateway_url(cli.gateway_url.as_deref())?;
            let api_key = std::env::var(&cli.api_key_env).with_context(|| {
                format!(
                    "reading embeddings API token from environment variable `{}`",
                    cli.api_key_env
                )
            })?;
            let client = GatewayEmbeddingsClient::new()?;
            let daemon = EmbeddingsDaemon::new(gateway_url, api_key, args.model, client);
            daemon.run_with_stdio(std::io::stdin().lock(), std::io::stdout())
        }
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .try_init();
}

fn resolve_gateway_url(explicit: Option<&str>) -> Result<String> {
    if let Some(gateway_url) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return Ok(gateway_url.to_string());
    }

    if let Some(base_url) = std::env::var(PLATFORM_GATEWAY_URL_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(format!("{}/v1/embeddings", base_url.trim_end_matches('/')));
    }

    Ok(DEFAULT_PLATFORM_EMBEDDINGS_GATEWAY_URL.to_string())
}

#[derive(Debug, Clone, PartialEq)]
struct EmbeddingsFailure {
    message: String,
}

trait EmbeddingsClient {
    fn embed(
        &self,
        gateway_url: &str,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingsFailure>;
}

struct GatewayEmbeddingsClient {
    client: reqwest::blocking::Client,
}

impl GatewayEmbeddingsClient {
    fn new() -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("building embeddings HTTP client")?;
        Ok(Self { client })
    }
}

impl EmbeddingsClient for GatewayEmbeddingsClient {
    fn embed(
        &self,
        gateway_url: &str,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingsFailure> {
        let payload = json!({
            "model": model,
            "input": texts,
        });

        let response = self
            .client
            .post(gateway_url)
            .bearer_auth(api_key)
            .json(&payload)
            .send()
            .map_err(|err| EmbeddingsFailure {
                message: err.to_string(),
            })?;

        let status = response.status();
        let body = response.json::<Value>().map_err(|err| EmbeddingsFailure {
            message: format!("gateway did not return JSON: {err}"),
        })?;

        if !status.is_success() {
            let message = body
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("embeddings gateway returned an error");
            return Err(EmbeddingsFailure {
                message: message.to_string(),
            });
        }

        extract_embedding_vectors(&body).map_err(|err| EmbeddingsFailure {
            message: err.to_string(),
        })
    }
}

struct EmbeddingsDaemon<C> {
    gateway_url: String,
    api_key: String,
    model: String,
    client: C,
}

impl<C> EmbeddingsDaemon<C>
where
    C: EmbeddingsClient,
{
    fn new(gateway_url: String, api_key: String, model: String, client: C) -> Self {
        Self {
            gateway_url,
            api_key,
            model,
            client,
        }
    }

    fn run_with_stdio<R, W>(&self, reader: R, mut writer: W) -> Result<()>
    where
        R: BufRead,
        W: Write,
    {
        write_json_line(
            &mut writer,
            &json!({
                "event": "ready",
                "protocol": READY_PROTOCOL_VERSION,
                "capabilities": CAPABILITIES
            }),
        )?;

        for line in reader.lines() {
            let line = line.context("reading embeddings runtime request")?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let request = match serde_json::from_str::<Value>(trimmed) {
                Ok(request) => request,
                Err(_) => {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "ok": false,
                            "error": {
                                "code": "INVALID_JSON",
                                "message": "could not parse request"
                            }
                        }),
                    )?;
                    continue;
                }
            };

            let should_stop =
                matches!(request.get("cmd").and_then(Value::as_str), Some("shutdown"));
            let response = self.handle_request(request);
            write_json_line(&mut writer, &response)?;
            if should_stop {
                break;
            }
        }

        Ok(())
    }

    fn handle_request(&self, request: Value) -> Value {
        if !request.is_object() {
            return error_response(None, "BAD_REQUEST", "request must be a JSON object");
        }

        let request_id = match request.get("id").and_then(Value::as_str) {
            Some(request_id) => request_id.to_string(),
            None => return error_response(None, "BAD_REQUEST", "id must be a string"),
        };
        let command = match request.get("cmd").and_then(Value::as_str) {
            Some(command) => command,
            None => {
                return error_response(Some(&request_id), "BAD_REQUEST", "cmd must be a string");
            }
        };

        if let Some(request_model) = request.get("model").and_then(Value::as_str)
            && request_model != self.model
        {
            return error_response(
                Some(&request_id),
                "BAD_REQUEST",
                &format!("model must match the daemon model: {}", self.model),
            );
        }

        match command {
            "ping" => json!({
                "id": request_id,
                "ok": true,
                "pong": true,
            }),
            "health" => json!({
                "id": request_id,
                "ok": true,
                "status": "ok",
                "model_loaded": true,
                "model": self.model,
            }),
            "embed" => self.handle_embed(&request_id, &request),
            "shutdown" => json!({
                "id": request_id,
                "ok": true,
            }),
            _ => error_response(
                Some(&request_id),
                "UNKNOWN_COMMAND",
                &format!("unsupported cmd: {command}"),
            ),
        }
    }

    fn handle_embed(&self, request_id: &str, request: &Value) -> Value {
        let Some(texts) = request.get("texts").and_then(Value::as_array) else {
            return error_response(
                Some(request_id),
                "BAD_REQUEST",
                "texts must be a non-empty array of strings",
            );
        };
        if texts.is_empty() || texts.iter().any(|text| text.as_str().is_none()) {
            return error_response(
                Some(request_id),
                "BAD_REQUEST",
                "texts must be a non-empty array of strings",
            );
        }

        let texts = texts
            .iter()
            .filter_map(|text| text.as_str().map(ToOwned::to_owned))
            .collect::<Vec<_>>();

        match self
            .client
            .embed(&self.gateway_url, &self.api_key, &self.model, &texts)
        {
            Ok(vectors) => json!({
                "id": request_id,
                "ok": true,
                "vectors": vectors,
                "model": self.model,
            }),
            Err(err) => error_response(Some(request_id), "INTERNAL", &err.message),
        }
    }
}

fn write_json_line<W>(writer: &mut W, payload: &Value) -> Result<()>
where
    W: Write,
{
    writeln!(
        writer,
        "{}",
        serde_json::to_string(payload).context("serialising JSON line")?
    )
    .context("writing JSON line")
}

fn error_response(request_id: Option<&str>, code: &str, message: &str) -> Value {
    let mut response = json!({
        "ok": false,
        "error": {
            "code": code,
            "message": message,
        }
    });
    if let Some(request_id) = request_id {
        response["id"] = Value::String(request_id.to_string());
    }
    response
}

fn extract_embedding_vectors(response: &Value) -> Result<Vec<Vec<f32>>> {
    let Some(rows) = response.get("data").and_then(Value::as_array) else {
        anyhow::bail!("embeddings response did not include a `data` array");
    };

    let mut vectors = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(embedding) = row.get("embedding").and_then(Value::as_array) else {
            anyhow::bail!("embeddings response row did not include an `embedding` array");
        };

        let mut vector = Vec::with_capacity(embedding.len());
        for value in embedding {
            let Some(number) = value.as_f64() else {
                anyhow::bail!("embedding value was not numeric");
            };
            if !number.is_finite() {
                anyhow::bail!("embedding value was not finite");
            }
            vector.push(number as f32);
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone)]
    struct StubClient {
        calls: Arc<Mutex<Vec<Vec<String>>>>,
        response: std::result::Result<Vec<Vec<f32>>, EmbeddingsFailure>,
    }

    impl StubClient {
        fn success(vectors: Vec<Vec<f32>>) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                response: Ok(vectors),
            }
        }

        fn failure(message: &str) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                response: Err(EmbeddingsFailure {
                    message: message.to_string(),
                }),
            }
        }
    }

    impl EmbeddingsClient for StubClient {
        fn embed(
            &self,
            gateway_url: &str,
            api_key: &str,
            model: &str,
            texts: &[String],
        ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingsFailure> {
            self.calls.lock().expect("calls mutex").push(vec![
                gateway_url.to_string(),
                api_key.to_string(),
                model.to_string(),
                texts.join("|"),
            ]);
            self.response.clone()
        }
    }

    fn fixture_value(name: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/protocol_fixtures")
            .join(name);
        serde_json::from_str(&fs::read_to_string(path).expect("fixture should load"))
            .expect("fixture should parse")
    }

    #[test]
    fn ready_event_matches_shared_fixture() {
        let daemon = EmbeddingsDaemon::new(
            "https://platform.example.com/v1/embeddings".to_string(),
            "secret".to_string(),
            "bge-m3".to_string(),
            StubClient::success(vec![vec![0.12, 0.98]]),
        );
        let input = Cursor::new(Vec::<u8>::new());
        let mut output = Vec::<u8>::new();

        daemon
            .run_with_stdio(input, &mut output)
            .expect("daemon should emit ready event");

        let first_line = String::from_utf8(output)
            .expect("utf-8 output")
            .lines()
            .next()
            .expect("ready line")
            .to_string();
        let actual: Value = serde_json::from_str(&first_line).expect("ready JSON");
        assert_eq!(actual, fixture_value("ready_event.json"));
    }

    #[test]
    fn ping_and_health_match_shared_fixtures() {
        let daemon = EmbeddingsDaemon::new(
            "https://platform.example.com/v1/embeddings".to_string(),
            "secret".to_string(),
            "bge-m3".to_string(),
            StubClient::success(vec![vec![0.12, 0.98]]),
        );
        let input = Cursor::new(
            [
                fs::read_to_string(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tests/protocol_fixtures/ping_request.json"),
                )
                .expect("ping request"),
                fs::read_to_string(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tests/protocol_fixtures/health_request.json"),
                )
                .expect("health request"),
                fs::read_to_string(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tests/protocol_fixtures/shutdown_request.json"),
                )
                .expect("shutdown request"),
            ]
            .join("\n")
                + "\n",
        );
        let mut output = Vec::<u8>::new();

        daemon
            .run_with_stdio(input, &mut output)
            .expect("daemon should handle requests");

        let messages = String::from_utf8(output)
            .expect("utf-8 output")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
            .collect::<Vec<_>>();

        assert_eq!(messages[0], fixture_value("ready_event.json"));
        assert_eq!(messages[1], fixture_value("ping_response.json"));
        assert_eq!(messages[2], fixture_value("health_response.json"));
        assert_eq!(messages[3], fixture_value("shutdown_response.json"));
    }

    #[test]
    fn embed_response_matches_shared_fixture() {
        let expected = fixture_value("embed_response.json");
        let vectors = expected
            .get("vectors")
            .and_then(Value::as_array)
            .expect("vectors")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("row")
                    .iter()
                    .map(|value| value.as_f64().expect("float") as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let client = StubClient::success(vectors);
        let daemon = EmbeddingsDaemon::new(
            "https://platform.example.com/v1/embeddings".to_string(),
            "secret".to_string(),
            "bge-m3".to_string(),
            client.clone(),
        );

        let response = daemon.handle_request(fixture_value("embed_request.json"));
        assert_eq!(response["id"], expected["id"]);
        assert_eq!(response["ok"], expected["ok"]);
        assert_eq!(response["model"], expected["model"]);

        let actual_rows = response["vectors"].as_array().expect("actual vectors");
        let expected_rows = expected["vectors"].as_array().expect("expected vectors");
        assert_eq!(actual_rows.len(), expected_rows.len());
        for (actual_row, expected_row) in actual_rows.iter().zip(expected_rows.iter()) {
            let actual_values = actual_row.as_array().expect("actual row");
            let expected_values = expected_row.as_array().expect("expected row");
            assert_eq!(actual_values.len(), expected_values.len());
            for (actual_value, expected_value) in actual_values.iter().zip(expected_values.iter()) {
                let actual_value = actual_value.as_f64().expect("actual float");
                let expected_value = expected_value.as_f64().expect("expected float");
                assert!(
                    (actual_value - expected_value).abs() < 0.000_001,
                    "expected {expected_value}, got {actual_value}"
                );
            }
        }

        let calls = client.calls.lock().expect("calls mutex");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0][0], "https://platform.example.com/v1/embeddings");
        assert_eq!(calls[0][1], "secret");
        assert_eq!(calls[0][2], "bge-m3");
        assert_eq!(calls[0][3], "hello|world");
    }

    #[test]
    fn gateway_errors_return_internal_protocol_errors() {
        let daemon = EmbeddingsDaemon::new(
            "https://platform.example.com/v1/embeddings".to_string(),
            "secret".to_string(),
            "bge-m3".to_string(),
            StubClient::failure("gateway timed out"),
        );

        let response = daemon.handle_request(json!({
            "id": "1",
            "cmd": "embed",
            "texts": ["hello"],
            "model": "bge-m3"
        }));

        assert_eq!(
            response,
            json!({
                "id": "1",
                "ok": false,
                "error": {
                    "code": "INTERNAL",
                    "message": "gateway timed out"
                }
            })
        );
    }

    #[test]
    fn invalid_json_lines_return_invalid_json_error() {
        let daemon = EmbeddingsDaemon::new(
            "https://platform.example.com/v1/embeddings".to_string(),
            "secret".to_string(),
            "bge-m3".to_string(),
            StubClient::success(vec![vec![0.12, 0.98]]),
        );
        let input = Cursor::new(b"not-json\n".to_vec());
        let mut output = Vec::<u8>::new();

        daemon
            .run_with_stdio(input, &mut output)
            .expect("daemon should respond to invalid JSON");

        let messages = String::from_utf8(output)
            .expect("utf-8 output")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
            .collect::<Vec<_>>();

        assert_eq!(messages[0], fixture_value("ready_event.json"));
        assert_eq!(
            messages[1],
            json!({
                "ok": false,
                "error": {
                    "code": "INVALID_JSON",
                    "message": "could not parse request"
                }
            })
        );
    }

    #[test]
    fn extracts_embedding_vectors_from_openai_response() {
        let response = json!({
            "data": [
                { "embedding": [1.0, 2.0] },
                { "embedding": [3.0, 4.0] }
            ]
        });
        assert_eq!(
            extract_embedding_vectors(&response).expect("vectors"),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }

    #[test]
    fn resolve_gateway_url_prefers_explicit_flag() {
        let url = resolve_gateway_url(Some("https://override.example/v1/embeddings"))
            .expect("explicit gateway url");
        assert_eq!(url, "https://override.example/v1/embeddings");
    }

    #[test]
    fn resolve_gateway_url_derives_from_platform_gateway_env() {
        unsafe {
            std::env::set_var(PLATFORM_GATEWAY_URL_ENV, "https://platform.example");
        }
        let url = resolve_gateway_url(None).expect("gateway url from env");
        unsafe {
            std::env::remove_var(PLATFORM_GATEWAY_URL_ENV);
        }
        assert_eq!(url, "https://platform.example/v1/embeddings");
    }

    #[test]
    fn resolve_gateway_url_defaults_to_production_gateway() {
        unsafe {
            std::env::remove_var(PLATFORM_GATEWAY_URL_ENV);
        }
        let url = resolve_gateway_url(None).expect("default gateway url");
        assert_eq!(url, DEFAULT_PLATFORM_EMBEDDINGS_GATEWAY_URL);
    }
}
