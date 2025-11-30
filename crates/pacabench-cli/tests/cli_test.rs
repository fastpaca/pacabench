use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;
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

    Command::new(assert_cmd::cargo::cargo_bin!("pacabench-cli"))
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
fn export_json_includes_judge_fields_and_precision() {
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
  "total_cases": 1,
  "completed_cases": 1,
  "start_time": "2024-01-01T00:00:00Z",
  "completed_time": "2024-01-01T00:00:01Z",
  "system_error_count": 0,
  "extras": {}
}"#,
    )
    .unwrap();

    // Single case with judge fields
    let result = serde_json::json!({
        "case_id": "1",
        "dataset_name": "ds",
        "agent_name": "agent",
        "passed": true,
        "output": "ok",
        "error": null,
        "error_type": "none",
        "runner_duration_ms": 10.0,
        "llm_metrics": { "llm_total_cost_usd": 0.5, "llm_call_count": 1 },
        "attempt": 1,
        "timestamp": "2024-01-01T00:00:00Z",
        "f1_score": 1.0,
        "f1_passed": true,
        "judge_passed": true,
        "judge_reason": "good",
        "judge_metrics": { "input_tokens": 10, "output_tokens": 5 },
        "judge_cost_usd": 1.0,
        "extra": {}
    });
    let mut results_file = fs::File::create(run_dir.join("results.jsonl")).unwrap();
    writeln!(results_file, "{}", result).unwrap();

    let output = Command::new(assert_cmd::cargo::cargo_bin!("pacabench-cli"))
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
    let metrics = &parsed["agents"]["agent"]["metrics"];
    assert_eq!(metrics["precision"], metrics["accuracy"]);
    assert_eq!(
        parsed["agents"]["agent"]["results"][0]["judge_cost_usd"]
            .as_f64()
            .unwrap(),
        1.0
    );
    assert_eq!(
        parsed["agents"]["agent"]["results"][0]["judge_metrics"]["input_tokens"]
            .as_u64()
            .unwrap(),
        10
    );
}
