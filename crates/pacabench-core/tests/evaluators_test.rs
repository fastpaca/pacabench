//! Tests for the evaluators module.

use pacabench_core::evaluators::{Evaluator, ExactMatchEvaluator, F1Evaluator, MultipleChoiceEvaluator};
use pacabench_core::types::{Case, ErrorType, RunnerOutput};
use std::collections::HashMap;

fn make_case(expected: &str) -> Case {
    Case {
        case_id: "1".into(),
        dataset_name: "ds".into(),
        input: "q".into(),
        expected: Some(expected.into()),
        history: vec![],
        metadata: HashMap::new(),
    }
}

fn ro(output: &str) -> RunnerOutput {
    RunnerOutput {
        output: Some(output.into()),
        error: None,
        metrics: None,
        duration_ms: 0.0,
        error_type: ErrorType::None,
        error_traceback: None,
        retry_count: 0,
    }
}

#[test]
fn exact_match_passes() {
    let ev = ExactMatchEvaluator;
    let res = ev.evaluate(&make_case("hi"), &ro("hi"));
    assert!(res.passed);
}

#[test]
fn f1_evaluator_scores() {
    let ev = F1Evaluator::new(0.5);
    let res = ev.evaluate(&make_case("hello world"), &ro("hello"));
    assert!(res.score > 0.0);
}

#[test]
fn multiple_choice_letter_match() {
    let mut case = make_case("A");
    let mut meta = serde_json::Map::new();
    meta.insert("A".into(), "foo".into());
    meta.insert("B".into(), "bar".into());
    let mut outer = HashMap::new();
    outer.insert("choices".into(), serde_json::Value::Object(meta));
    case.metadata = outer;
    let ev = MultipleChoiceEvaluator;
    let res = ev.evaluate(&case, &ro("A"));
    assert!(res.passed);
}
