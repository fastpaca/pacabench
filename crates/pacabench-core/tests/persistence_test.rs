//! Tests for the persistence module.

use pacabench_core::config::{Config, ConfigOverrides};
use pacabench_core::persistence::{
    compute_config_fingerprint, iso_timestamp_now, list_run_summaries, ErrorEntry, RunMetadata,
    RunStore,
};
use pacabench_core::types::{CaseResult, ErrorType, LlmMetrics, RunStatus};
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn fingerprint_is_stable() {
    let cfg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples/membench_qa_test/pacabench.yaml");
    let cfg = Config::from_file(cfg_path, ConfigOverrides::default()).expect("config");
    let fp1 = compute_config_fingerprint(&cfg).unwrap();
    let fp2 = compute_config_fingerprint(&cfg).unwrap();
    assert_eq!(fp1, fp2);
}

#[test]
fn run_store_roundtrip_results_and_errors() {
    let dir = tempdir().unwrap();
    let store = RunStore::new(dir.path()).unwrap();

    let result = CaseResult {
        case_id: "1".into(),
        dataset_name: "ds".into(),
        agent_name: "agent".into(),
        passed: true,
        output: Some("ok".into()),
        error: None,
        error_type: ErrorType::None,
        runner_duration_ms: 10.0,
        llm_metrics: LlmMetrics::default(),
        attempt: 1,
        timestamp: Some(iso_timestamp_now()),
        f1_score: None,
        f1_passed: None,
        judge_passed: None,
        judge_reason: None,
        judge_metrics: None,
    };

    store.append_result(&result).unwrap();

    let error_entry = ErrorEntry {
        timestamp: iso_timestamp_now(),
        error_type: ErrorType::SystemFailure,
        agent_name: Some("agent".into()),
        dataset_name: Some("ds".into()),
        case_id: Some("1".into()),
        error: Some("boom".into()),
    };

    store.append_error(&error_entry).unwrap();

    let loaded_results = store.load_results().unwrap();
    assert_eq!(loaded_results.len(), 1);
    assert_eq!(loaded_results[0].output.as_deref(), Some("ok"));

    let loaded_errors = store.load_errors().unwrap();
    assert_eq!(loaded_errors.len(), 1);
    assert_eq!(loaded_errors[0].error.as_deref(), Some("boom"));
}

#[test]
fn list_run_summaries_includes_progress() {
    let dir = tempdir().unwrap();
    let store1 = RunStore::new(dir.path().join("run-a")).unwrap();
    let meta1 = RunMetadata::new(
        "run-a".into(),
        "fp".into(),
        vec!["agent".into()],
        vec!["ds".into()],
        10,
    );
    let mut meta1 = meta1;
    meta1.status = RunStatus::Completed;
    meta1.completed_cases = 10;
    meta1.start_time = Some("2024-01-01T00:00:00Z".into());
    meta1.completed_time = Some("2024-01-01T00:10:00Z".into());
    store1.write_metadata(&meta1).unwrap();

    let store2 = RunStore::new(dir.path().join("run-b")).unwrap();
    let mut meta2 = RunMetadata::new(
        "run-b".into(),
        "fp".into(),
        vec!["agent".into()],
        vec!["ds".into()],
        8,
    );
    meta2.status = RunStatus::Running;
    meta2.completed_cases = 4;
    meta2.start_time = Some("2024-02-01T00:00:00Z".into());
    meta2.system_error_count = 1;
    store2.write_metadata(&meta2).unwrap();

    let summaries = list_run_summaries(dir.path()).unwrap();
    assert_eq!(summaries.len(), 2);
    assert_eq!(summaries[0].run_id, "run-b");
    assert!((summaries[0].progress - 0.5).abs() < 0.01);
    assert_eq!(summaries[1].completed_cases, 10);
}
