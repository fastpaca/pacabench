//! OpenAI-compatible proxy server for metrics collection (basic).
//! Tracks request bodies for now; extend to capture usage/latency/cost.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use reqwest::Client;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
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
) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let model = body.get("model").and_then(|v| v.as_str()).map(String::from);

    if let Some(base) = &state.upstream_base {
        let url = format!("{base}{path}");
        let mut req = state.client.post(&url).json(&body);
        if let Some(key) = &state.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
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
                return (status, Json(json));
            }
            Err(err) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({ "error": err.to_string() })),
                );
            }
        }
    }

    let latency_ms = start.elapsed().as_millis() as f64;
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
    (StatusCode::OK, Json(response))
}
