//! OpenAI-compatible proxy server for metrics collection.

use crate::error::{PacabenchError, Result};
use crate::types::LlmMetrics;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use parking_lot::Mutex;
use reqwest::Client;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tracing::info;

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub port: u16,
    pub upstream_base_url: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Default)]
pub struct MetricsCollector {
    call_count: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    cached_tokens: AtomicU64,
    total_latency_us: AtomicU64,
    latencies: Mutex<Vec<f64>>,
    model: Mutex<Option<String>>,
}

impl MetricsCollector {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn record(&self, latency_ms: f64, usage: &Value, model: Option<String>) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us
            .fetch_add((latency_ms * 1000.0) as u64, Ordering::Relaxed);

        self.latencies.lock().push(latency_ms);

        if let Some(u) = usage.as_object() {
            if let Some(v) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.input_tokens.fetch_add(v, Ordering::Relaxed);
            }
            if let Some(v) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens.fetch_add(v, Ordering::Relaxed);
            }
            if let Some(v) = u
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_u64())
            {
                self.cached_tokens.fetch_add(v, Ordering::Relaxed);
            }
        }

        if let Some(m) = model {
            *self.model.lock() = Some(m);
        }
    }

    pub fn snapshot_and_clear(&self) -> LlmMetrics {
        let call_count = self.call_count.swap(0, Ordering::Relaxed);
        let input_tokens = self.input_tokens.swap(0, Ordering::Relaxed);
        let output_tokens = self.output_tokens.swap(0, Ordering::Relaxed);
        let cached_tokens = self.cached_tokens.swap(0, Ordering::Relaxed);
        let _ = self.total_latency_us.swap(0, Ordering::Relaxed);

        let latencies = std::mem::take(&mut *self.latencies.lock());
        let model = self.model.lock().take();

        LlmMetrics {
            call_count,
            latency_ms: latencies,
            input_tokens,
            output_tokens,
            cached_tokens,
            model,
        }
    }
}

#[derive(Clone)]
struct ProxyState {
    metrics: Arc<MetricsCollector>,
    upstream_base: String,
    api_key: Option<String>,
    client: Client,
}

pub struct ProxyServer {
    handle: JoinHandle<()>,
    pub addr: SocketAddr,
    pub metrics: Arc<MetricsCollector>,
}

impl ProxyServer {
    pub async fn start(cfg: ProxyConfig) -> Result<Self> {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], cfg.port)))
            .await
            .map_err(PacabenchError::Proxy)?;
        let addr = listener.local_addr().map_err(PacabenchError::Proxy)?;
        let metrics = MetricsCollector::new();

        let state = ProxyState {
            metrics: metrics.clone(),
            upstream_base: cfg.upstream_base_url,
            api_key: cfg.api_key,
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

    pub fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }
}

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
    forward_request(&state, headers, body, "/v1/chat/completions").await
}

async fn handle_responses(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    forward_request(&state, headers, body, "/v1/responses").await
}

async fn handle_embeddings(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    forward_request(&state, headers, body, "/v1/embeddings").await
}

async fn forward_request(
    state: &ProxyState,
    headers: HeaderMap,
    mut body: Value,
    path: &str,
) -> Response {
    let start = std::time::Instant::now();
    let request_model = body.get("model").and_then(|v| v.as_str()).map(String::from);

    // Force non-streaming for simplicity
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_string(), Value::Bool(false));
        obj.remove("stream_options");
    }

    let url = format!("{}{}", state.upstream_base, path);
    let mut req = state.client.post(&url).json(&body);

    if let Some(key) = &state.api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    } else if let Some(auth) = headers.get(header::AUTHORIZATION) {
        req = req.header(header::AUTHORIZATION, auth.clone());
    }

    for (name, value) in headers.iter() {
        if name == header::AUTHORIZATION || name == header::HOST || name == header::CONTENT_LENGTH {
            continue;
        }
        let name_str = name.as_str().to_lowercase();
        if name_str == "api-key"
            || name_str == "x-api-key"
            || name_str == "openai-organization"
            || name_str == "azure-api-key"
        {
            req = req.header(name, value.clone());
        }
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let json = resp
                .json::<Value>()
                .await
                .unwrap_or_else(|_| serde_json::json!({"error": "invalid json"}));

            let latency_ms = start.elapsed().as_millis() as f64;
            let usage = json.get("usage").cloned().unwrap_or_default();
            let response_model = json
                .get("model")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or(request_model);

            state.metrics.record(latency_ms, &usage, response_model);
            (status, Json(json)).into_response()
        }
        Err(err) => {
            let latency_ms = start.elapsed().as_millis() as f64;
            state
                .metrics
                .record(latency_ms, &Value::Null, request_model);

            let body = serde_json::json!({"error": err.to_string()});
            (StatusCode::BAD_GATEWAY, Json(body)).into_response()
        }
    }
}
