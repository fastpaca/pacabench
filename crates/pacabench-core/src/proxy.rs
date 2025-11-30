//! OpenAI-compatible proxy server for metrics collection.
//!
//! The proxy intercepts OpenAI API calls, forwards them to the upstream provider,
//! and records metrics (latency, token usage, cost) for each request.
//!
//! # Event-Driven Metrics
//!
//! The proxy can emit events for each LLM request via an [`EventSender`]. This
//! allows real-time progress tracking and cost monitoring. When an event sender
//! is configured, metrics are emitted immediately; otherwise they are buffered
//! for later retrieval via `snapshot_and_clear()`.

use crate::events::{EventSender, PacabenchEvent};
use crate::pricing::calculate_cost;
use axum::{
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use parking_lot::Mutex;
use reqwest::Client;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::info;

/// Configuration for the proxy server.
#[derive(Clone, Debug)]
pub struct ProxyConfig {
    /// Port to listen on. If zero, an ephemeral port is chosen.
    pub port: u16,
    /// Base URL of the upstream API (e.g., "https://api.openai.com").
    pub upstream_base_url: Option<String>,
    /// API key for authentication.
    pub api_key: Option<String>,
}

/// Collects metrics from LLM requests.
///
/// Supports two modes:
/// - **Buffered**: Metrics are stored and retrieved via `snapshot_and_clear()`
/// - **Event-driven**: Metrics are emitted immediately via an `EventSender`
#[derive(Clone, Debug, Default)]
pub struct MetricsCollector {
    inner: Arc<Mutex<Vec<MetricEntry>>>,
    event_sender: Option<EventSender>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a metrics collector that emits events.
    pub fn with_events(event_sender: EventSender) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Vec::new())),
            event_sender: Some(event_sender),
        }
    }

    /// Record a metric entry.
    ///
    /// If an event sender is configured, emits an `LlmRequest` event.
    /// Otherwise, buffers the entry for later retrieval.
    pub fn record(&self, entry: MetricEntry) {
        // Emit event if configured
        if let Some(sender) = &self.event_sender {
            let (input_tokens, output_tokens, cached_tokens) = extract_tokens(&entry.usage);
            let cost = entry
                .model
                .as_ref()
                .map(|m| calculate_cost(m, input_tokens, output_tokens, cached_tokens))
                .unwrap_or(0.0);

            sender.emit(PacabenchEvent::LlmRequest {
                latency_ms: entry.latency_ms,
                model: entry.model.clone(),
                input_tokens,
                output_tokens,
                cached_tokens,
                cost_usd: cost,
            });
        }

        // Always buffer for backward compatibility
        self.inner.lock().push(entry);
    }

    /// Get all recorded metrics and clear the buffer.
    pub fn snapshot_and_clear(&self) -> Vec<MetricEntry> {
        let mut guard = self.inner.lock();
        std::mem::take(&mut *guard)
    }

    /// Get the current count of recorded metrics.
    pub fn count(&self) -> usize {
        self.inner.lock().len()
    }
}

/// Extract token counts from a usage JSON value.
fn extract_tokens(usage: &Value) -> (u64, u64, u64) {
    let input = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    (input, output, cached)
}

/// A single metric entry from an LLM request.
#[derive(Clone, Debug)]
pub struct MetricEntry {
    /// Request latency in milliseconds.
    pub latency_ms: f64,
    /// Token usage information from the response.
    pub usage: Value,
    /// HTTP status code of the response.
    pub status: StatusCode,
    /// Model used for the request.
    pub model: Option<String>,
}

/// Internal state shared across request handlers.
#[derive(Clone)]
struct ProxyState {
    metrics: MetricsCollector,
    upstream_base: Option<String>,
    api_key: Option<String>,
    client: Client,
}

/// OpenAI-compatible proxy server.
///
/// Intercepts API requests, forwards them upstream, and collects metrics.
/// The server is automatically stopped when dropped (RAII).
pub struct ProxyServer {
    handle: JoinHandle<()>,
    /// The address the proxy is listening on.
    pub addr: SocketAddr,
    /// Metrics collector for this proxy.
    pub metrics: MetricsCollector,
}

impl ProxyServer {
    /// Start a proxy server with the given configuration.
    pub async fn start(cfg: ProxyConfig) -> anyhow::Result<Self> {
        Self::start_with_events(cfg, None).await
    }

    /// Start a proxy server that emits events.
    pub async fn start_with_events(
        cfg: ProxyConfig,
        event_sender: Option<EventSender>,
    ) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], cfg.port))).await?;
        let addr = listener.local_addr()?;

        let metrics = match event_sender {
            Some(sender) => MetricsCollector::with_events(sender),
            None => MetricsCollector::new(),
        };

        let state = ProxyState {
            metrics: metrics.clone(),
            upstream_base: cfg.upstream_base_url.clone(),
            api_key: cfg.api_key.clone(),
            client: Client::new(),
        };

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat))
            .route("/v1/responses", post(handle_responses))
            .route("/v1/embeddings", post(handle_embeddings))
            .route("/health", get(|| async { "ok" }))
            .with_state(state);

        let handle = tokio::spawn(async move {
            info!("proxy listening on {}", addr);
            if let Err(e) = axum::serve(listener, app).await {
                tracing::warn!("proxy server error: {e}");
            }
        });

        Ok(Self {
            handle,
            addr,
            metrics,
        })
    }

    /// Get the base URL for this proxy.
    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }
}

/// RAII cleanup: abort the server task when dropped.
impl Drop for ProxyServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn handle_chat(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    forward_or_stub(&state, headers, body, "/v1/chat/completions").await
}

async fn handle_responses(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    forward_or_stub(&state, headers, body, "/v1/responses").await
}

async fn handle_embeddings(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    forward_or_stub(&state, headers, body, "/v1/embeddings").await
}

async fn forward_or_stub(
    state: &ProxyState,
    headers: HeaderMap,
    body: Value,
    path: &str,
) -> Response {
    let start = std::time::Instant::now();
    let model = body.get("model").and_then(|v| v.as_str()).map(String::from);
    let wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Some(base) = &state.upstream_base {
        let url = format!("{base}{path}");
        let mut req = state.client.post(&url).json(&body);
        if let Some(key) = &state.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        } else if let Some(auth) = headers.get(header::AUTHORIZATION) {
            req = req.header(header::AUTHORIZATION, auth.clone());
        }
        for (name, value) in headers.iter() {
            if name == header::AUTHORIZATION
                || name == header::HOST
                || name == header::CONTENT_LENGTH
            {
                continue;
            }
            if name.as_str().eq_ignore_ascii_case("api-key")
                || name.as_str().eq_ignore_ascii_case("x-api-key")
                || name.as_str().eq_ignore_ascii_case("openai-organization")
                || name.as_str().eq_ignore_ascii_case("azure-api-key")
            {
                req = req.header(name, value.clone());
            }
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();

                if wants_stream {
                    return stream_response(resp, state.metrics.clone(), model, start, status);
                }

                let json = resp
                    .json::<Value>()
                    .await
                    .unwrap_or_else(|_| serde_json::json!({ "error": "invalid json" }));
                let latency_ms = start.elapsed().as_millis() as f64;
                let usage = json.get("usage").cloned().unwrap_or_else(default_usage);
                // Response may contain model info that differs from request
                let response_model = json
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or(model);
                state.metrics.record(MetricEntry {
                    latency_ms,
                    usage,
                    status,
                    model: response_model,
                });
                return (status, Json(json)).into_response();
            }
            Err(err) => {
                let body = serde_json::json!({ "error": err.to_string() });
                return (StatusCode::BAD_GATEWAY, Json(body)).into_response();
            }
        }
    }

    let latency_ms = start.elapsed().as_millis() as f64;
    if wants_stream {
        return stub_stream(latency_ms, model, state.metrics.clone());
    }

    state.metrics.record(MetricEntry {
        latency_ms,
        usage: default_usage(),
        status: StatusCode::OK,
        model,
    });
    let response = serde_json::json!({
        "id": "stub",
        "object": "chat.completion",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "stubbed"
            }
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    });
    (StatusCode::OK, Json(response)).into_response()
}

fn stub_stream(latency_ms: f64, model: Option<String>, metrics: MetricsCollector) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(4);
    let m_for_record = model.clone();
    tokio::spawn(async move {
        let first = format!(
            "data: {{\"model\":\"{}\",\"choices\":[{{\"delta\":{{\"content\":\"stub\"}}}}]}}\n\n",
            model.clone().unwrap_or_else(|| "stub-model".into())
        );
        let _ = tx.send(Ok(axum::body::Bytes::from(first))).await;
        let _ = tx
            .send(Ok(axum::body::Bytes::from_static(b"data: [DONE]\n\n")))
            .await;
        metrics.record(MetricEntry {
            latency_ms,
            usage: serde_json::json!({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0}
            }),
            status: StatusCode::OK,
            model: m_for_record,
        });
    });
    let stream = ReceiverStream::new(rx);
    build_sse_response(stream)
}

fn stream_response(
    resp: reqwest::Response,
    metrics: MetricsCollector,
    initial_model: Option<String>,
    start: std::time::Instant,
    status: StatusCode,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(16);
    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut usage_totals = (0u64, 0u64, 0u64);
        let mut saw_usage = false;
        let mut last_model = initial_model;

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        for line in text.lines() {
                            let line = line.trim_start();
                            if let Some(data) = line.strip_prefix("data:") {
                                let data = data.trim();
                                if data == "[DONE]" {
                                    continue;
                                }
                                if let Ok(json) = serde_json::from_str::<Value>(data) {
                                    if let Some(m) = json.get("model").and_then(|v| v.as_str()) {
                                        last_model = Some(m.to_string());
                                    }
                                    if let Some(u) = json.get("usage") {
                                        let prompt = u
                                            .get("prompt_tokens")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                        let completion = u
                                            .get("completion_tokens")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                        let cached = u
                                            .get("prompt_tokens_details")
                                            .and_then(|d| d.get("cached_tokens"))
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0);
                                        usage_totals.0 += prompt;
                                        usage_totals.1 += completion;
                                        usage_totals.2 += cached;
                                        saw_usage = true;
                                    }
                                }
                            }
                        }
                    }
                    let _ = tx.send(Ok(bytes)).await;
                }
                Err(_) => break,
            }
        }

        let latency_ms = start.elapsed().as_millis() as f64;
        let usage = if saw_usage {
            serde_json::json!({
                "prompt_tokens": usage_totals.0,
                "completion_tokens": usage_totals.1,
                "total_tokens": usage_totals.0 + usage_totals.1,
                "prompt_tokens_details": {"cached_tokens": usage_totals.2}
            })
        } else {
            default_usage()
        };
        metrics.record(MetricEntry {
            latency_ms,
            usage,
            status,
            model: last_model,
        });
    });

    let stream = ReceiverStream::new(rx);
    build_sse_response(stream)
}

/// Build a server-sent events response.
fn build_sse_response(
    stream: ReceiverStream<Result<axum::body::Bytes, std::io::Error>>,
) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )
        .body(axum::body::Body::from_stream(stream))
        .expect("valid SSE response")
}

fn default_usage() -> Value {
    serde_json::json!({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0}
    })
}
