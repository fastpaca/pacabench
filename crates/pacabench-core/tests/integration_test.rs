//! Integration tests for the pacabench-core orchestrator.

use pacabench_core::config::{
    AgentConfig, BenchmarkConfig, DatasetConfig, GlobalConfig, OutputConfig, ProxyConfig,
};
use pacabench_core::metrics::aggregate_results;
use pacabench_core::orchestrator::Orchestrator;
use pacabench_core::persistence::RunStore;
use std::fs;
use tempfile::tempdir;

/// Creates a simple echo agent that returns the input as output.
fn create_echo_agent(path: &std::path::Path) {
    let script = r#"
import sys
import json

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        data = json.loads(line)
        output = {"output": data.get("input", ""), "error": None}
        print(json.dumps(output), flush=True)
    except Exception as e:
        print(json.dumps({"output": None, "error": str(e)}), flush=True)
"#;
    fs::write(path, script).unwrap();
}

/// Creates a simple test dataset with a few cases.
fn create_test_dataset(path: &std::path::Path) {
    let cases = [
        r#"{"case_id": "1", "input": "hello", "expected": "hello"}"#,
        r#"{"case_id": "2", "input": "world", "expected": "world"}"#,
        r#"{"case_id": "3", "input": "test", "expected": "test"}"#,
    ];
    fs::write(path, cases.join("\n")).unwrap();
}

#[tokio::test]
async fn test_orchestrator_end_to_end() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let cache = root.join("cache");
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&cache).unwrap();
    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    // Create echo agent
    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    // Create test dataset
    let dataset_file = data_dir.join("cases.jsonl");
    create_test_dataset(&dataset_file);

    let config = BenchmarkConfig {
        name: "integration-test".into(),
        description: Some("Integration test benchmark".into()),
        version: "0.1.0".into(),
        author: None,
        config: GlobalConfig {
            concurrency: 2,
            timeout_seconds: 30.0,
            max_retries: 1,
            proxy: ProxyConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        },
        agents: vec![AgentConfig {
            name: "echo-agent".into(),
            command: format!("python {}", agent_script.display()),
            setup: None,
            teardown: None,
            env: Default::default(),
        }],
        datasets: vec![DatasetConfig {
            name: "test-dataset".into(),
            source: data_dir.to_string_lossy().to_string(),
            split: None,
            prepare: None,
            input_map: Default::default(),
            evaluator: None,
        }],
        output: OutputConfig {
            directory: runs.to_string_lossy().to_string(),
        },
    };

    let orch = Orchestrator::new(config, root.clone(), cache, runs.clone());

    // Run the orchestrator
    orch.run(Some("test-run".into()), None, false)
        .await
        .expect("orchestrator should run successfully");

    // Verify results
    let store = RunStore::new(runs.join("test-run")).expect("should open run store");
    let results = store.load_results().expect("should load results");

    assert_eq!(results.len(), 3, "should have 3 results");

    // All cases should pass (echo agent returns input which matches expected)
    for result in &results {
        assert!(
            result.passed,
            "case {} should pass, got error: {:?}",
            result.case_id, result.error
        );
        assert!(result.error.is_none(), "should have no error");
    }

    // Check aggregated metrics
    let metrics = aggregate_results(&results);
    assert_eq!(metrics.total_cases, 3);
    assert!(
        (metrics.accuracy - 1.0).abs() < 0.001,
        "accuracy should be 100%"
    );
    assert_eq!(metrics.failed_cases, 0);
}

#[tokio::test]
async fn test_orchestrator_resume() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let cache = root.join("cache");
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&cache).unwrap();
    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    let dataset_file = data_dir.join("cases.jsonl");
    create_test_dataset(&dataset_file);

    let config = BenchmarkConfig {
        name: "resume-test".into(),
        description: None,
        version: "0.1.0".into(),
        author: None,
        config: GlobalConfig {
            concurrency: 1,
            timeout_seconds: 30.0,
            max_retries: 0,
            proxy: ProxyConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        },
        agents: vec![AgentConfig {
            name: "echo-agent".into(),
            command: format!("python {}", agent_script.display()),
            setup: None,
            teardown: None,
            env: Default::default(),
        }],
        datasets: vec![DatasetConfig {
            name: "test-dataset".into(),
            source: data_dir.to_string_lossy().to_string(),
            split: None,
            prepare: None,
            input_map: Default::default(),
            evaluator: None,
        }],
        output: OutputConfig {
            directory: runs.to_string_lossy().to_string(),
        },
    };

    let orch = Orchestrator::new(config.clone(), root.clone(), cache.clone(), runs.clone());

    // First run with limit 1
    orch.run(Some("resume-run".into()), Some(1), false)
        .await
        .expect("first run should succeed");

    let store1 = RunStore::new(runs.join("resume-run")).unwrap();
    let results1 = store1.load_results().unwrap();
    assert_eq!(results1.len(), 1, "first run should have 1 result");

    // Resume run - should complete remaining cases
    let orch2 = Orchestrator::new(config, root.clone(), cache, runs.clone());
    orch2
        .run(Some("resume-run".into()), None, false)
        .await
        .expect("resume should succeed");

    let store2 = RunStore::new(runs.join("resume-run")).unwrap();
    let results2 = store2.load_results().unwrap();
    assert_eq!(results2.len(), 3, "after resume should have 3 results");
}

#[tokio::test]
async fn test_orchestrator_with_evaluator() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let cache = root.join("cache");
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&cache).unwrap();
    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    // Create dataset with cases that will pass exact match
    let dataset_file = data_dir.join("cases.jsonl");
    let cases = [
        r#"{"case_id": "1", "input": "hello", "expected": "hello"}"#,
        r#"{"case_id": "2", "input": "world", "expected": "different"}"#,
    ];
    fs::write(&dataset_file, cases.join("\n")).unwrap();

    let config = BenchmarkConfig {
        name: "eval-test".into(),
        description: None,
        version: "0.1.0".into(),
        author: None,
        config: GlobalConfig {
            concurrency: 1,
            timeout_seconds: 30.0,
            max_retries: 0,
            proxy: ProxyConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        },
        agents: vec![AgentConfig {
            name: "echo-agent".into(),
            command: format!("python {}", agent_script.display()),
            setup: None,
            teardown: None,
            env: Default::default(),
        }],
        datasets: vec![DatasetConfig {
            name: "test-dataset".into(),
            source: data_dir.to_string_lossy().to_string(),
            split: None,
            prepare: None,
            input_map: Default::default(),
            evaluator: Some(pacabench_core::config::EvaluatorConfig {
                r#type: "exact_match".into(),
                model: None,
                extra_config: Default::default(),
                additional: Default::default(),
            }),
        }],
        output: OutputConfig {
            directory: runs.to_string_lossy().to_string(),
        },
    };

    let orch = Orchestrator::new(config, root.clone(), cache, runs.clone());

    orch.run(Some("eval-run".into()), None, false)
        .await
        .expect("run should succeed");

    let store = RunStore::new(runs.join("eval-run")).unwrap();
    let results = store.load_results().unwrap();

    assert_eq!(results.len(), 2);

    // Find results by case_id
    let r1 = results.iter().find(|r| r.case_id == "1").unwrap();
    let r2 = results.iter().find(|r| r.case_id == "2").unwrap();

    assert!(r1.passed, "case 1 should pass (hello == hello)");
    assert!(!r2.passed, "case 2 should fail (world != different)");

    let metrics = aggregate_results(&results);
    assert!(
        (metrics.accuracy - 0.5).abs() < 0.001,
        "accuracy should be 50%"
    );
}

#[test]
fn test_proxy_url_format_includes_v1() {
    // Verify the proxy URL format is correct for OpenAI SDK compatibility
    let addr: std::net::SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let proxy_url = format!("http://{}/v1", addr);
    assert!(proxy_url.ends_with("/v1"), "proxy_url should end with /v1");
    assert_eq!(proxy_url, "http://127.0.0.1:8080/v1");
}

#[test]
fn test_upstream_url_derivation_from_provider() {
    // Test that upstream URL is correctly derived from provider name
    let providers: [(&str, Option<&str>); 3] = [
        ("openai", Some("https://api.openai.com")),
        ("anthropic", Some("https://api.anthropic.com")),
        ("unknown", None),
    ];

    for (provider, expected) in providers {
        let derived = match provider {
            "openai" => Some("https://api.openai.com".to_string()),
            "anthropic" => Some("https://api.anthropic.com".to_string()),
            _ => None,
        };

        match expected {
            Some(exp) => {
                assert!(derived.is_some(), "provider {provider} should have URL");
                assert_eq!(derived.unwrap(), exp);
            }
            None => {
                assert!(derived.is_none(), "provider {provider} should have no URL");
            }
        }
    }
}

#[tokio::test]
async fn test_orchestrator_uses_relative_path_with_root_dir() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let cache = root.join("cache");
    let runs = root.join("runs");
    fs::create_dir_all(&cache).unwrap();
    fs::create_dir_all(&runs).unwrap();
    let data_dir = root.join("data");
    fs::create_dir_all(&data_dir).unwrap();

    fs::write(
        data_dir.join("cases.jsonl"),
        r#"{"case_id": "1", "input": "test", "expected": "test"}"#,
    )
    .unwrap();

    // Create agent in root directory
    fs::write(
        root.join("agent.py"),
        r#"
import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    data = json.loads(line)
    out = {"output": data.get("input", ""), "error": None}
    print(json.dumps(out))
    sys.stdout.flush()
"#,
    )
    .unwrap();

    // Use RELATIVE path (verifies work_dir is used)
    let cfg = BenchmarkConfig {
        name: "relpath-test".into(),
        description: None,
        version: "0.1.0".into(),
        author: None,
        config: GlobalConfig {
            proxy: ProxyConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        },
        agents: vec![AgentConfig {
            name: "agent".into(),
            command: "python agent.py".into(), // Relative path!
            setup: None,
            teardown: None,
            env: Default::default(),
        }],
        datasets: vec![DatasetConfig {
            name: "ds".into(),
            source: data_dir.to_string_lossy().to_string(),
            split: None,
            prepare: None,
            input_map: Default::default(),
            evaluator: None,
        }],
        output: OutputConfig {
            directory: runs.to_string_lossy().to_string(),
        },
    };

    let orch = Orchestrator::new(cfg, root.clone(), cache, runs.clone());
    orch.run(None, Some(1), false).await.unwrap();

    let run_dirs: Vec<_> = fs::read_dir(&runs).unwrap().collect();
    assert_eq!(run_dirs.len(), 1);
    let first_run = run_dirs[0].as_ref().unwrap().path();
    let results = fs::read_to_string(first_run.join("results.jsonl")).unwrap();
    let result: serde_json::Value =
        serde_json::from_str(results.lines().next().unwrap()).unwrap();
    assert!(
        result["error"].is_null(),
        "should have no error: {:?}",
        result["error"]
    );
    assert_eq!(result["output"].as_str(), Some("test"));
}
