//! Tests for the proxy module.

use pacabench_core::proxy::{ProxyConfig, ProxyServer};
use reqwest::Client;

#[tokio::test]
async fn proxy_starts_and_provides_stub_without_upstream() {
    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: None,
        api_key: None,
    })
    .await
    .unwrap();

    let client = reqwest::Client::new();
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

    let metrics = proxy.metrics.snapshot_and_clear().await;
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].model.as_deref(), Some("gpt-4"));

    proxy.stop().await;
}

#[tokio::test]
async fn proxy_records_metrics() {
    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: None,
        api_key: None,
    })
    .await
    .unwrap();

    let client = reqwest::Client::new();

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

    let metrics = proxy.metrics.snapshot_and_clear().await;
    assert_eq!(metrics.len(), 3);

    let metrics2 = proxy.metrics.snapshot_and_clear().await;
    assert!(metrics2.is_empty());

    proxy.stop().await;
}

#[tokio::test]
async fn proxy_url_should_include_v1_prefix() {
    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: None,
        api_key: None,
    })
    .await
    .unwrap();

    let expected_url = format!("http://{}/v1", proxy.addr);
    assert!(expected_url.ends_with("/v1"));

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/chat/completions", expected_url))
        .json(&serde_json::json!({"model": "test", "messages": []}))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    proxy.stop().await;
}

#[tokio::test]
async fn proxy_healthcheck() {
    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: None,
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
    proxy.stop().await;
}

#[tokio::test]
async fn proxy_streaming_stub_records_metric() {
    let proxy = ProxyServer::start(ProxyConfig {
        port: 0,
        upstream_base_url: None,
        api_key: None,
    })
    .await
    .unwrap();

    let resp = Client::new()
        .post(format!("http://{}/v1/chat/completions", proxy.addr))
        .json(&serde_json::json!({
            "model": "stream-model",
            "messages": [],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("data:"),
        "streaming response should contain data lines"
    );

    let metrics = proxy.metrics.snapshot_and_clear().await;
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].model.as_deref(), Some("stream-model"));
    proxy.stop().await;
}
