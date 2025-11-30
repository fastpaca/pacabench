//! OpenAI-compatible proxy server for metrics collection (basic).
//! Tracks request bodies for now; extend to capture usage/latency/cost.

use axum::{
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::info;

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    /// If zero, an ephemeral port is chosen.
    pub port: u16,
    pub upstream_base_url: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct MetricsCollector {
    inner: Arc<Mutex<Vec<MetricEntry>>>,
}

impl MetricsCollector {
    pub async fn record(&self, entry: MetricEntry) {
        let mut guard = self.inner.lock().await;
        guard.push(entry);
    }

    pub async fn snapshot_and_clear(&self) -> Vec<MetricEntry> {
        let mut guard = self.inner.lock().await;
        let out = guard.clone();
        guard.clear();
        out
    }
}

#[derive(Clone, Debug)]
pub struct MetricEntry {
    pub latency_ms: f64,
    pub usage: Value,
    pub status: StatusCode,
    pub model: Option<String>,
}

#[derive(Clone)]
struct ProxyState {
    metrics: MetricsCollector,
    upstream_base: Option<String>,
    api_key: Option<String>,
    client: Client,
}

pub struct ProxyServer {
    handle: JoinHandle<()>,
    pub addr: SocketAddr,
    pub metrics: MetricsCollector,
}

impl ProxyServer {
    pub async fn start(cfg: ProxyConfig) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], cfg.port))).await?;
        let addr = listener.local_addr()?;
        let state = ProxyState {
            metrics: MetricsCollector::default(),
            upstream_base: cfg.upstream_base_url.clone(),
            api_key: cfg.api_key.clone(),
            client: Client::new(),
        };

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat))
            .route("/v1/responses", post(handle_responses))
            .route("/v1/embeddings", post(handle_embeddings))
            .route("/health", get(|| async { "ok" }))
            .with_state(state.clone());

        let metrics = state.metrics.clone();
        let handle = tokio::spawn(async move {
            info!("proxy listening on {}", addr);
            axum::serve(listener, app).await.ok();
        });

        Ok(Self {
            handle,
            addr,
            metrics,
        })
    }

    pub async fn stop(self) {
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
    _headers: HeaderMap,
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
                let usage = json.get("usage").cloned().unwrap_or_default();
                // Response may contain model info that differs from request
                let response_model = json
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or(model);
                state
                    .metrics
                    .record(MetricEntry {
                        latency_ms,
                        usage,
                        status,
                        model: response_model,
                    })
                    .await;
                return (status, Json(json)).into_response();
            }
            Err(err) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({ "error": err.to_string() })),
                )
                    .into_response();
            }
        }
    }

    let latency_ms = start.elapsed().as_millis() as f64;
    if wants_stream {
        return stub_stream(latency_ms, model, state.metrics.clone());
    }

    state
        .metrics
        .record(MetricEntry {
            latency_ms,
            usage: serde_json::json!({}),
            status: StatusCode::OK,
            model,
        })
        .await;
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
            "total_tokens": 0
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
        metrics
            .record(MetricEntry {
                latency_ms,
                usage: serde_json::json!({
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 0}
                }),
                status: StatusCode::OK,
                model: m_for_record,
            })
            .await;
    });
    let stream = ReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
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
        let mut last_usage: Option<Value> = None;
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
                                        last_usage = Some(u.clone());
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
        metrics
            .record(MetricEntry {
                latency_ms,
                usage: last_usage.unwrap_or_else(|| serde_json::json!({})),
                status,
                model: last_model,
            })
            .await;
    });

    let stream = ReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}
