//! Tests for the metrics module.

use pacabench_core::metrics::aggregate_results;
use pacabench_core::types::{CaseResult, ErrorType};
use serde_json::json;
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

#[test]
fn aggregates_judge_tokens_separately() {
    let mut r1 = make_result(5.0, true);
    r1.judge_metrics.insert("input_tokens".into(), json!(10));
    r1.judge_metrics.insert("output_tokens".into(), json!(2));
    r1.judge_cost_usd = Some(0.5);

    let mut r2 = make_result(7.0, false);
    r2.llm_metrics
        .insert("llm_total_cost_usd".into(), json!(1.0));
    r2.judge_metrics.insert("input_tokens".into(), json!(3));

    let m = aggregate_results(&[r1, r2]);
    assert_eq!(
        m.total_cost_usd, 1.0,
        "judge cost should not be in model cost"
    );
    assert_eq!(m.total_judge_cost_usd, 0.5);
    assert_eq!(m.total_judge_input_tokens, 13);
    assert_eq!(m.total_judge_output_tokens, 2);
}
