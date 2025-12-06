//! Tests for the metrics module.

use pacabench_core::metrics::aggregate_results;
use pacabench_core::types::{CaseResult, ErrorType, JudgeMetrics, LlmMetrics};

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
        llm_metrics: LlmMetrics::default(),
        attempt: 1,
        timestamp: None,
        f1_score: None,
        f1_passed: None,
        judge_passed: None,
        judge_reason: None,
        judge_metrics: None,
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
fn aggregates_judge_tokens_separately() {
    let mut r1 = make_result(5.0, true);
    r1.judge_metrics = Some(JudgeMetrics {
        input_tokens: 10,
        output_tokens: 2,
        cached_tokens: 0,
        latency_ms: 0.0,
        model: None,
    });

    let mut r2 = make_result(7.0, false);
    r2.llm_metrics = LlmMetrics {
        call_count: 1,
        latency_ms: vec![],
        input_tokens: 100,
        output_tokens: 50,
        cached_tokens: 0,
        model: None,
    };
    r2.judge_metrics = Some(JudgeMetrics {
        input_tokens: 3,
        output_tokens: 0,
        cached_tokens: 0,
        latency_ms: 0.0,
        model: None,
    });

    let m = aggregate_results(&[r1, r2]);
    // Judge tokens should be aggregated separately from LLM tokens
    assert_eq!(m.total_input_tokens, 100);
    assert_eq!(m.total_output_tokens, 50);
    assert_eq!(m.total_judge_input_tokens, 13);
    assert_eq!(m.total_judge_output_tokens, 2);
}

#[test]
fn aggregates_attempts() {
    let mut r1 = make_result(5.0, true);
    r1.attempt = 1;
    let mut r2 = make_result(7.0, false);
    r2.attempt = 3;
    let m = aggregate_results(&[r1, r2]);
    assert_eq!(m.total_attempts, 4);
    assert!((m.avg_attempts - 2.0).abs() < 1e-6);
    assert_eq!(m.max_attempts, 3);
}
