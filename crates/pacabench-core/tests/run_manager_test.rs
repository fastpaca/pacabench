//! Tests for the run_manager module.

use pacabench_core::config::{BenchmarkConfig, GlobalConfig, OutputConfig};
use pacabench_core::persistence::iso_timestamp_now;
use pacabench_core::run_manager::RunManager;
use pacabench_core::types::{CaseResult, ErrorType};
use tempfile::tempdir;

fn minimal_config() -> BenchmarkConfig {
    BenchmarkConfig {
        name: "test-bench".to_string(),
        description: None,
        version: "0.1.0".to_string(),
        author: None,
        config: GlobalConfig::default(),
        agents: vec![],
        datasets: vec![],
        output: OutputConfig::default(),
    }
}

fn sample_result(attempt: u32) -> CaseResult {
    CaseResult {
        case_id: "1".into(),
        dataset_name: "ds".into(),
        agent_name: "agent".into(),
        passed: true,
        output: Some("ok".into()),
        error: None,
        error_type: ErrorType::None,
        runner_duration_ms: 10.0,
        llm_metrics: Default::default(),
        attempt,
        timestamp: Some(iso_timestamp_now()),
        f1_score: None,
        f1_passed: None,
        judge_passed: None,
        judge_reason: None,
        judge_metrics: Default::default(),
        judge_cost_usd: None,
        extra: Default::default(),
    }
}

#[test]
fn attempts_increment_and_resume() {
    let cfg = minimal_config();
    let runs_dir = tempdir().unwrap();

    #[allow(unused_mut)]
    let mut rm = RunManager::new(&cfg, runs_dir.path().to_path_buf(), None, false).unwrap();
    rm.set_total_cases(5).unwrap();
    rm.initialize_metadata().unwrap();

    assert_eq!(rm.get_attempt_count("agent", "ds", "1"), 0);
    let attempt1 = rm.get_next_attempt("agent", "ds", "1");
    assert_eq!(attempt1, 1);
    let mut res = sample_result(attempt1);
    rm.append_result(&res).unwrap();
    assert_eq!(rm.get_attempt_count("agent", "ds", "1"), 1);

    let attempt2 = rm.get_next_attempt("agent", "ds", "1");
    res.attempt = attempt2;
    rm.append_result(&res).unwrap();
    assert_eq!(rm.get_attempt_count("agent", "ds", "1"), 2);

    // Resume with same run_id
    let rm2 = RunManager::new(
        &cfg,
        runs_dir.path().to_path_buf(),
        Some(rm.run_id.clone()),
        false,
    )
    .unwrap();
    assert!(rm2.resuming);
    assert_eq!(rm2.get_attempt_count("agent", "ds", "1"), 2);
    assert_eq!(rm2.completed_count(), 1);
}

#[test]
fn fingerprint_mismatch_errors() {
    let cfg = minimal_config();
    let runs_dir = tempdir().unwrap();
    let rm = RunManager::new(&cfg, runs_dir.path().to_path_buf(), None, false).unwrap();
    rm.initialize_metadata().unwrap();

    // Different config fingerprint triggers error unless force_new
    let mut cfg2 = cfg.clone();
    cfg2.name = "other".into();
    let err = RunManager::new(
        &cfg2,
        runs_dir.path().to_path_buf(),
        Some(rm.run_id.clone()),
        false,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("different config fingerprint"));
}
