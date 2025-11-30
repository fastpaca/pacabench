//! Tests for the proxy module.

use axum::{routing::post, Json, Router};
use pacabench_core::proxy::{ProxyConfig, ProxyServer};
use reqwest::Client;
use tokio::net::TcpListener;

async fn start_mock_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}", addr);

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Json(serde_json::json!({
                "id": "test",
                "model": "gpt-4",
                "choices": [{"message": {"role": "assistant", "content": "test"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            }))
        }),
    );

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    (url, handle)
}

#[tokio::test]
async fn proxy_starts_and_forwards_requests() {
    let (upstream_url, _handle) = start_mock_upstream().await;

    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: upstream_url,
        api_key: None,
    })
    .await
    .unwrap();

    let client = Client::new();
    let resp = client
        .post(format!("http://{}/v1/chat/completions", proxy.addr))
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}]
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.get("choices").is_some());

    let metrics = proxy.metrics.snapshot_and_clear();
    assert_eq!(metrics.call_count, 1);
    assert_eq!(metrics.input_tokens, 5);
    assert_eq!(metrics.output_tokens, 2);
}

#[tokio::test]
async fn proxy_records_metrics() {
    let (upstream_url, _handle) = start_mock_upstream().await;

    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: upstream_url,
        api_key: None,
    })
    .await
    .unwrap();

    let client = Client::new();

    for _ in 0..3 {
        client
            .post(format!("http://{}/v1/chat/completions", proxy.addr))
            .json(&serde_json::json!({
                "model": "test-model",
                "messages": []
            }))
            .send()
            .await
            .unwrap();
    }

    let metrics = proxy.metrics.snapshot_and_clear();
    assert_eq!(metrics.call_count, 3);

    let metrics2 = proxy.metrics.snapshot_and_clear();
    assert_eq!(metrics2.call_count, 0);
}

#[tokio::test]
async fn proxy_url_should_include_v1_prefix() {
    let (upstream_url, _handle) = start_mock_upstream().await;

    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: upstream_url,
        api_key: None,
    })
    .await
    .unwrap();

    let expected_url = format!("http://{}/v1", proxy.addr);
    assert!(expected_url.ends_with("/v1"));

    let client = Client::new();
    let resp = client
        .post(format!("{}/chat/completions", expected_url))
        .json(&serde_json::json!({"model": "test", "messages": []}))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
}

#[tokio::test]
async fn proxy_healthcheck() {
    let (upstream_url, _handle) = start_mock_upstream().await;

    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: upstream_url,
        api_key: None,
    })
    .await
    .unwrap();

    let resp = Client::new()
        .get(format!("http://{}/health", proxy.addr))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
}
