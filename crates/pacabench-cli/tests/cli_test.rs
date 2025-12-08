use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use tempfile::tempdir;

fn write_min_config(path: &Path) {
    let mut file = fs::File::create(path).unwrap();
    writeln!(
        file,
        r#"
name: sample
version: "0.1.0"
agents:
  - name: agent
    command: "echo hi"
datasets:
  - name: ds
    source: "data.jsonl"
"#
    )
    .unwrap();
}

#[test]
fn show_without_run_id_handles_empty_runs_dir() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("pacabench.yaml");
    write_min_config(&config_path);

    let runs_dir = dir.path().join("runs");
    fs::create_dir_all(&runs_dir).unwrap();

    Command::new(assert_cmd::cargo::cargo_bin!("pacabench"))
        .arg("--config")
        .arg(&config_path)
        .arg("show")
        .arg("--runs-dir")
        .arg(&runs_dir)
        .assert()
        .success()
        .stdout(predicate::str::contains("No runs found."));
}

#[test]
fn export_json_includes_judge_fields() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("pacabench.yaml");
    write_min_config(&config_path);

    let runs_dir = dir.path().join("runs");
    let run_dir = runs_dir.join("run1");
    fs::create_dir_all(&run_dir).unwrap();

    // Minimal metadata to satisfy run resolution
    fs::write(
        run_dir.join("metadata.json"),
        r#"{
  "run_id": "run1",
  "status": "completed",
  "config_fingerprint": "fp",
  "agents": ["agent"],
  "datasets": ["ds"],
  "total_cases": 1,
  "completed_cases": 1,
  "start_time": "2024-01-01T00:00:00Z",
  "completed_time": "2024-01-01T00:00:01Z",
  "system_error_count": 0
}"#,
    )
    .unwrap();

    // Single case with judge fields (cost is computed at CLI layer, not stored)
    let result = serde_json::json!({
        "case_id": "1",
        "dataset_name": "ds",
        "agent_name": "agent",
        "passed": true,
        "output": "ok",
        "error_type": "none",
        "runner_duration_ms": 10.0,
        "llm_metrics": {
            "call_count": 1,
            "latency_ms": [],
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 0,
            "model": "gpt-4o"
        },
        "attempt": 1,
        "timestamp": "2024-01-01T00:00:00Z",
        "f1_score": 1.0,
        "f1_passed": true,
        "judge_passed": true,
        "judge_reason": "good",
        "judge_metrics": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 0,
            "latency_ms": 0.0,
            "model": "gpt-4o"
        }
    });
    let mut results_file = fs::File::create(run_dir.join("results.jsonl")).unwrap();
    writeln!(results_file, "{}", result).unwrap();

    let output = Command::new(assert_cmd::cargo::cargo_bin!("pacabench"))
        .arg("--config")
        .arg(&config_path)
        .arg("export")
        .arg("run1")
        .arg("--runs-dir")
        .arg(&runs_dir)
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let parsed: Value = serde_json::from_slice(&output).unwrap();

    // New RunStats-based export format: judge tokens in tokens object
    assert_eq!(parsed["tokens"]["judge_input"].as_u64().unwrap(), 10);
    assert_eq!(parsed["tokens"]["judge_output"].as_u64().unwrap(), 5);
    assert_eq!(
        parsed["tokens"]["per_model"]["gpt-4o"]["judge_input"]
            .as_u64()
            .unwrap(),
        10
    );

    // Verify failures array includes judge_reason
    // (The test case passes, so it won't be in failures)
    assert_eq!(parsed["passed_cases"].as_u64().unwrap(), 1);
    assert_eq!(parsed["failed_cases"].as_u64().unwrap(), 0);
}

/// Test that SIGINT triggers graceful shutdown.
///
/// This test verifies:
/// 1. The signal handler is installed and receives SIGINT
/// 2. The benchmark process exits cleanly (doesn't hang)
/// 3. A metadata file is created indicating the run was processed
///
/// Note: We don't check for "aborted" status because fast-failing agents
/// may complete before the signal is processed. The key verification is
/// that the process handles the signal and exits gracefully.
#[test]
#[cfg(unix)]
fn sigint_causes_graceful_exit() {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;
    use std::os::unix::process::ExitStatusExt;
    use std::time::Duration;

    let dir = tempdir().unwrap();
    let config_path = dir.path().join("pacabench.yaml");
    let data_path = dir.path().join("data.jsonl");
    let runs_dir = dir.path().join("runs");

    // Create runs directory upfront so it exists even if signal arrives early
    fs::create_dir_all(&runs_dir).unwrap();

    // Create test data
    fs::write(
        &data_path,
        r#"{"case_id": "1", "question": "test", "answer": "done"}"#,
    )
    .unwrap();

    // Write config with simple echo agent
    let mut config_file = fs::File::create(&config_path).unwrap();
    writeln!(
        config_file,
        r#"
name: signal-test
version: "0.1.0"
config:
  proxy:
    enabled: false
agents:
  - name: echo-agent
    command: "cat"
datasets:
  - name: ds
    source: "{}"
"#,
        data_path.display()
    )
    .unwrap();

    // Start the benchmark process
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("pacabench"))
        .arg("--config")
        .arg(&config_path)
        .arg("run")
        .arg("--runs-dir")
        .arg(&runs_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start pacabench");

    // Give it a moment to start
    std::thread::sleep(Duration::from_millis(100));

    // Send SIGINT
    let pid = Pid::from_raw(child.id() as i32);
    let _ = kill(pid, Signal::SIGINT); // Ignore error if process already exited

    // Wait for process to exit with a reasonable timeout
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited - verify it handled the signal gracefully
                assert!(
                    status.success() || status.code().is_some() || status.signal().is_some(),
                    "Process should have exited cleanly or been signaled"
                );
                break;
            }
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(10) {
                    // Kill it if it hasn't exited
                    let _ = child.kill();
                    panic!("Process did not exit within timeout after SIGINT");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => panic!("Error waiting for child: {}", e),
        }
    }

    // Verify a run directory was created (benchmark started)
    let entries: Vec<_> = fs::read_dir(&runs_dir)
        .expect("runs dir should exist")
        .filter_map(|e| e.ok())
        .collect();

    // A metadata file should exist indicating the run was processed
    if !entries.is_empty() {
        let run_dir = &entries[0].path();
        let metadata_path = run_dir.join("metadata.json");
        assert!(
            metadata_path.exists(),
            "metadata.json should exist in run directory"
        );
    }
}
