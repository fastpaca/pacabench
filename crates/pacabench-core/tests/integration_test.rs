//! Integration tests for the pacabench-core Benchmark API.

use pacabench_core::config::{
    AgentConfig, Config, DatasetConfig, EvaluatorConfig, GlobalConfig, OutputConfig, ProxyConfig,
};
use pacabench_core::metrics::aggregate_results;
use pacabench_core::persistence::RunStore;
use pacabench_core::Benchmark;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

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

fn create_test_dataset(path: &std::path::Path) {
    let cases = [
        r#"{"case_id": "1", "input": "hello", "expected": "hello"}"#,
        r#"{"case_id": "2", "input": "world", "expected": "world"}"#,
        r#"{"case_id": "3", "input": "test", "expected": "test"}"#,
    ];
    fs::write(path, cases.join("\n")).unwrap();
}

fn make_test_config(
    name: &str,
    agent_command: String,
    data_source: String,
    root_dir: PathBuf,
    runs_dir: PathBuf,
    evaluator: Option<EvaluatorConfig>,
) -> Config {
    Config {
        name: name.into(),
        description: None,
        version: "0.1.0".into(),
        author: None,
        global: GlobalConfig {
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
            command: agent_command,
            setup: None,
            teardown: None,
            env: Default::default(),
        }],
        datasets: vec![DatasetConfig {
            name: "test-dataset".into(),
            source: data_source,
            split: None,
            prepare: None,
            input_map: Default::default(),
            evaluator,
        }],
        output: OutputConfig {
            directory: runs_dir.to_string_lossy().to_string(),
        },
        root_dir: root_dir.clone(),
        cache_dir: root_dir.join("cache"),
        runs_dir,
        config_path: None,
    }
}

#[tokio::test]
async fn test_benchmark_end_to_end() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    let dataset_file = data_dir.join("cases.jsonl");
    create_test_dataset(&dataset_file);

    let config = make_test_config(
        "integration-test",
        format!("python {}", agent_script.display()),
        data_dir.to_string_lossy().to_string(),
        root.clone(),
        runs.clone(),
        None,
    );

    let bench = Benchmark::new(config);
    let result = bench
        .run(Some("test-run".into()), None)
        .await
        .expect("benchmark should run successfully");

    assert!(!result.aborted, "run should not be aborted");

    let store = RunStore::new(runs.join("test-run")).expect("should open run store");
    let results = store.load_results().expect("should load results");

    assert_eq!(results.len(), 3, "should have 3 results");

    for result in &results {
        assert!(
            result.passed,
            "case {} should pass, got error: {:?}",
            result.case_id, result.error
        );
        assert!(result.error.is_none(), "should have no error");
    }

    let metrics = aggregate_results(&results);
    assert_eq!(metrics.total_cases, 3);
    assert!(
        (metrics.accuracy - 1.0).abs() < 0.001,
        "accuracy should be 100%"
    );
    assert_eq!(metrics.failed_cases, 0);
}

/// Test resume of an interrupted (non-terminal) run.
///
/// This simulates a run that was interrupted mid-execution and needs to be resumed.
/// The key difference from retry is that resume processes ALL remaining cases,
/// not just failed ones from the original run.
#[tokio::test]
async fn test_benchmark_resume() {
    use pacabench_core::types::RunStatus;

    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    let dataset_file = data_dir.join("cases.jsonl");
    create_test_dataset(&dataset_file);

    let mut config = make_test_config(
        "resume-test",
        format!("python {}", agent_script.display()),
        data_dir.to_string_lossy().to_string(),
        root.clone(),
        runs.clone(),
        None,
    );
    config.global.concurrency = 1;
    config.global.max_retries = 0;

    // First run with limit 1
    let bench1 = Benchmark::new(config.clone());
    bench1
        .run(Some("resume-run".into()), Some(1))
        .await
        .expect("first run should succeed");

    let store1 = RunStore::new(runs.join("resume-run")).unwrap();
    let results1 = store1.load_results().unwrap();
    assert_eq!(results1.len(), 1, "first run should have 1 result");

    // Simulate an interrupted run by setting status back to Running
    let mut metadata = store1.read_metadata().unwrap().unwrap();
    metadata.status = RunStatus::Running;
    store1.write_metadata(&metadata).unwrap();

    // Resume run - should process remaining cases since run is non-terminal
    let bench2 = Benchmark::new(config);
    bench2
        .run(Some("resume-run".into()), None)
        .await
        .expect("resume should succeed");

    let store2 = RunStore::new(runs.join("resume-run")).unwrap();
    let results2 = store2.load_results().unwrap();
    assert_eq!(results2.len(), 3, "after resume should have 3 results");
}

#[tokio::test]
async fn test_benchmark_with_evaluator() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    let dataset_file = data_dir.join("cases.jsonl");
    let cases = [
        r#"{"case_id": "1", "input": "hello", "expected": "hello"}"#,
        r#"{"case_id": "2", "input": "world", "expected": "different"}"#,
    ];
    fs::write(&dataset_file, cases.join("\n")).unwrap();

    let mut config = make_test_config(
        "eval-test",
        format!("python {}", agent_script.display()),
        data_dir.to_string_lossy().to_string(),
        root.clone(),
        runs.clone(),
        Some(EvaluatorConfig::ExactMatch),
    );
    config.global.concurrency = 1;
    config.global.max_retries = 0;

    let bench = Benchmark::new(config);
    bench
        .run(Some("eval-run".into()), None)
        .await
        .expect("run should succeed");

    let store = RunStore::new(runs.join("eval-run")).unwrap();
    let results = store.load_results().unwrap();

    assert_eq!(results.len(), 2);

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
    let addr: std::net::SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let proxy_url = format!("http://{}/v1", addr);
    assert!(proxy_url.ends_with("/v1"), "proxy_url should end with /v1");
    assert_eq!(proxy_url, "http://127.0.0.1:8080/v1");
}

#[test]
fn test_upstream_url_derivation_from_provider() {
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
async fn test_benchmark_uses_relative_path_with_root_dir() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let runs = root.join("runs");
    fs::create_dir_all(&runs).unwrap();
    let data_dir = root.join("data");
    fs::create_dir_all(&data_dir).unwrap();

    fs::write(
        data_dir.join("cases.jsonl"),
        r#"{"case_id": "1", "input": "test", "expected": "test"}"#,
    )
    .unwrap();

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

    let config = Config {
        name: "relpath-test".into(),
        description: None,
        version: "0.1.0".into(),
        author: None,
        global: GlobalConfig {
            proxy: ProxyConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        },
        agents: vec![AgentConfig {
            name: "agent".into(),
            command: "python agent.py".into(),
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
        root_dir: root.clone(),
        cache_dir: root.join("cache"),
        runs_dir: runs.clone(),
        config_path: None,
    };

    let bench = Benchmark::new(config);
    bench.run(None, Some(1)).await.unwrap();

    let run_dirs: Vec<_> = fs::read_dir(&runs).unwrap().collect();
    assert_eq!(run_dirs.len(), 1);
    let first_run = run_dirs[0].as_ref().unwrap().path();
    let results = fs::read_to_string(first_run.join("results.jsonl")).unwrap();
    let result: serde_json::Value = serde_json::from_str(results.lines().next().unwrap()).unwrap();
    assert!(
        result["error"].is_null(),
        "should have no error: {:?}",
        result["error"]
    );
    assert_eq!(result["output"].as_str(), Some("test"));
}

#[tokio::test]
async fn worker_pool_routes_cases_to_matching_agent() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_a = root.join("agent_a.py");
    let agent_b = root.join("agent_b.py");

    let agent_script = |path: &std::path::Path, label: &str| {
        let script = format!(
            r#"
import sys, json

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        json.loads(line)
        print(json.dumps({{"output": "{label}", "error": None}}), flush=True)
    except Exception as e:
        print(json.dumps({{"output": None, "error": str(e)}}), flush=True)
"#
        );
        fs::write(path, script).unwrap();
    };

    agent_script(&agent_a, "agent-a");
    agent_script(&agent_b, "agent-b");

    let dataset_file = data_dir.join("cases.jsonl");
    let cases = [
        r#"{"case_id":"1","input":"foo"}"#,
        r#"{"case_id":"2","input":"bar"}"#,
    ];
    fs::write(&dataset_file, cases.join("\n")).unwrap();

    let config = Config {
        name: "agent-routing".into(),
        description: None,
        version: "0.1.0".into(),
        author: None,
        global: GlobalConfig {
            concurrency: 4,
            timeout_seconds: 5.0,
            max_retries: 0,
            proxy: ProxyConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        },
        agents: vec![
            AgentConfig {
                name: "agent-a".into(),
                command: format!("python {}", agent_a.display()),
                setup: None,
                teardown: None,
                env: Default::default(),
            },
            AgentConfig {
                name: "agent-b".into(),
                command: format!("python {}", agent_b.display()),
                setup: None,
                teardown: None,
                env: Default::default(),
            },
        ],
        datasets: vec![DatasetConfig {
            name: "routing-ds".into(),
            source: data_dir.to_string_lossy().to_string(),
            split: None,
            prepare: None,
            input_map: Default::default(),
            evaluator: None,
        }],
        output: OutputConfig {
            directory: runs.to_string_lossy().to_string(),
        },
        root_dir: root.clone(),
        cache_dir: root.join("cache"),
        runs_dir: runs.clone(),
        config_path: None,
    };

    let bench = Benchmark::new(config);
    bench.run(None, Some(2)).await.unwrap();

    let run_dirs: Vec<_> = fs::read_dir(&runs).unwrap().collect();
    assert_eq!(run_dirs.len(), 1);
    let run_dir = run_dirs[0].as_ref().unwrap().path();
    let results = RunStore::new(run_dir).unwrap().load_results().unwrap();

    assert_eq!(results.len(), 4);
    for result in results {
        if result.agent_name == "agent-a" {
            assert_eq!(result.output.as_deref(), Some("agent-a"));
            assert!(
                result.error.is_none(),
                "agent-a should not error: {:?}",
                result.error
            );
        } else if result.agent_name == "agent-b" {
            assert_eq!(result.output.as_deref(), Some("agent-b"));
            assert!(
                result.error.is_none(),
                "agent-b should not error: {:?}",
                result.error
            );
        } else {
            panic!("unexpected agent {}", result.agent_name);
        }
    }
}

/// Test that retry only retries cases from the original run, not new cases from the dataset.
///
/// This tests the bug where:
/// 1. Run with --limit 2 on a 5-case dataset (case 1 passes, case 2 fails)
/// 2. Retry the run
/// 3. Bug: All 5 cases are retried instead of just the 1 failed case from the original run
#[tokio::test]
async fn test_retry_only_retries_failed_cases_from_original_run() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let runs = root.join("runs");
    let data_dir = root.join("data");

    fs::create_dir_all(&runs).unwrap();
    fs::create_dir_all(&data_dir).unwrap();

    let agent_script = root.join("agent.py");
    create_echo_agent(&agent_script);

    // Create dataset with 5 cases, where case 2 will fail (expected != input)
    let dataset_file = data_dir.join("cases.jsonl");
    let cases = [
        r#"{"case_id": "1", "input": "hello", "expected": "hello"}"#,
        r#"{"case_id": "2", "input": "world", "expected": "DIFFERENT"}"#,
        r#"{"case_id": "3", "input": "foo", "expected": "foo"}"#,
        r#"{"case_id": "4", "input": "bar", "expected": "bar"}"#,
        r#"{"case_id": "5", "input": "baz", "expected": "baz"}"#,
    ];
    fs::write(&dataset_file, cases.join("\n")).unwrap();

    let mut config = make_test_config(
        "retry-limit-test",
        format!("python {}", agent_script.display()),
        data_dir.to_string_lossy().to_string(),
        root.clone(),
        runs.clone(),
        Some(EvaluatorConfig::ExactMatch),
    );
    config.global.concurrency = 1;
    config.global.max_retries = 0; // No automatic retries

    // First run with limit 2: runs cases 1 and 2 only
    let bench1 = Benchmark::new(config.clone());
    bench1
        .run(Some("retry-test-run".into()), Some(2))
        .await
        .expect("first run should succeed");

    let store1 = RunStore::new(runs.join("retry-test-run")).unwrap();
    let results1 = store1.load_results().unwrap();

    assert_eq!(results1.len(), 2, "first run should have exactly 2 results");

    let passed_count = results1.iter().filter(|r| r.passed).count();
    let failed_count = results1.iter().filter(|r| !r.passed).count();
    assert_eq!(passed_count, 1, "should have 1 passed case");
    assert_eq!(failed_count, 1, "should have 1 failed case");

    // Retry: should only retry the 1 failed case, not run cases 3,4,5
    let bench2 = Benchmark::new(config);

    // Subscribe to events to check that total_cases is reported correctly
    let mut events = bench2.subscribe();

    let result2 = bench2
        .run(Some("retry-test-run".into()), None)
        .await
        .expect("retry should succeed");

    let store2 = RunStore::new(runs.join("retry-test-run")).unwrap();
    let results2 = store2.load_results().unwrap();

    // Verify only cases from original run are processed
    assert_eq!(
        results2.len(),
        2,
        "retry should only process cases from original run, not new cases. \
         Got {} results but expected 2. Case IDs: {:?}",
        results2.len(),
        results2.iter().map(|r| &r.case_id).collect::<Vec<_>>()
    );

    // Verify total_cases in RunStarted event reflects only the failed cases being retried
    // Not the full dataset (5 cases) or the original run (2 cases)
    // This is what the progress display shows to the user
    let mut found_run_started = false;
    while let Ok(event) = events.try_recv() {
        if let pacabench_core::Event::RunStarted { total_cases, .. } = event {
            assert_eq!(
                total_cases, 1,
                "RunStarted.total_cases should be 1 (only failed case), not full dataset"
            );
            found_run_started = true;
            break;
        }
    }
    assert!(found_run_started, "should have received RunStarted event");

    // Verify run completed successfully
    assert!(!result2.aborted, "run should not be aborted");
}
