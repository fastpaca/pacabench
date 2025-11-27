//! Tests for the metrics module.

use pacabench_core::metrics::aggregate_results;
use pacabench_core::types::{CaseResult, ErrorType};
use std::collections::HashMap;

fn make_result(ms: f64, passed: bool) -> CaseResult {
    CaseResult {
        case_id: "1".into(),
        dataset_name: "ds".into(),
        agent_name: "agent".into(),
        passed,
        output: Some("o".into()),
        error: None,
        error_type: ErrorType::None,
        runner_duration_ms: ms,
        llm_metrics: HashMap::new(),
        attempt: 1,
        timestamp: None,
        f1_score: None,
        f1_passed: None,
        judge_passed: None,
        judge_reason: None,
        judge_metrics: HashMap::new(),
        judge_cost_usd: None,
        extra: HashMap::new(),
    }
}

#[test]
fn aggregates_accuracy() {
    let res = vec![make_result(10.0, true), make_result(20.0, false)];
    let m = aggregate_results(&res);
    assert_eq!(m.total_cases, 2);
    assert_eq!(m.failed_cases, 1);
    assert!((m.accuracy - 0.5).abs() < 1e-6);
}

#[test]
fn precision_matches_accuracy_when_only_pass_fail_known() {
    let res = vec![make_result(5.0, true), make_result(8.0, true)];
    let m = aggregate_results(&res);
    assert!((m.precision - m.accuracy).abs() < 1e-6);
    assert!((m.precision - 1.0).abs() < 1e-6);
}
