//! Tests for the evaluators module.

use axum::{extract::State, routing::post, Json, Router};
use pacabench_core::config::MultipleChoiceFallback;
use pacabench_core::evaluators::{
    Evaluator, ExactMatchEvaluator, F1Evaluator, LlmJudgeEvaluator, MultipleChoiceEvaluator,
};
use pacabench_core::runner::RunnerOutput;
use pacabench_core::types::{Case, ErrorType};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

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
        error_type: ErrorType::None,
        duration_ms: 0.0,
        error_traceback: None,
    }
}

#[tokio::test]
async fn exact_match_passes() {
    let ev = ExactMatchEvaluator;
    let res = ev.evaluate(&make_case("hi"), &ro("hi")).await;
    assert!(res.passed);
}

#[tokio::test]
async fn f1_evaluator_scores() {
    let ev = F1Evaluator::new(0.5);
    let res = ev.evaluate(&make_case("hello world"), &ro("hello")).await;
    assert!(res.score > 0.0);
}

#[tokio::test]
async fn f1_evaluator_handles_punctuation() {
    let ev = F1Evaluator::new(0.5);
    let res = ev
        .evaluate(&make_case("Hello world"), &ro("Hello, world!"))
        .await;
    assert!(res.passed, "punctuation should not break token match");
}

#[tokio::test]
async fn multiple_choice_letter_match() {
    let mut case = make_case("A");
    let mut meta = serde_json::Map::new();
    meta.insert("A".into(), "foo".into());
    meta.insert("B".into(), "bar".into());
    let mut outer = HashMap::new();
    outer.insert("choices".into(), serde_json::Value::Object(meta));
    case.metadata = outer;

    let ev =
        MultipleChoiceEvaluator::new(MultipleChoiceFallback::F1, 0.5, "gpt-4o-mini".to_string());
    let res = ev.evaluate(&case, &ro("A")).await;
    assert!(res.passed);
}

#[tokio::test]
async fn multiple_choice_fallbacks_to_f1_without_choices() {
    let mut case = make_case("foo bar");
    case.metadata = HashMap::new();

    let ev =
        MultipleChoiceEvaluator::new(MultipleChoiceFallback::F1, 0.5, "gpt-4o-mini".to_string());
    let res = ev.evaluate(&case, &ro("foo bar")).await;
    assert!(res.passed, "fallback F1 should be used when no choices");
}

#[tokio::test(flavor = "multi_thread")]
async fn llm_judge_records_tokens_and_cost() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Json(serde_json::json!({
                "choices": [{"message": {"content": "YES"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}}
            }))
        }),
    );
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    std::env::set_var("OPENAI_API_KEY", "test");
    std::env::set_var("JUDGE_BASE_URL", format!("http://{}", addr));

    let judge = LlmJudgeEvaluator::new("gpt-4o-mini".into(), None, None, 1, 0);
    let res = judge.evaluate(&make_case("hello"), &ro("hello")).await;
    handle.abort();
    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("JUDGE_BASE_URL");
    assert!(res.passed);
    let jm = res.judge_metrics.expect("judge_metrics should be present");
    assert_eq!(jm.input_tokens, 10);
    assert_eq!(jm.output_tokens, 2);
    assert!(jm.latency_ms > 0.0, "latency should be recorded");
    assert!(
        jm.model.is_some(),
        "model should be recorded for cost calculation"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn llm_judge_retries_and_uses_api_key_and_base_url() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let last_auth: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    #[derive(Clone)]
    struct AppState {
        calls: Arc<AtomicUsize>,
        last_auth: Arc<Mutex<Option<String>>>,
    }

    let state = AppState {
        calls: calls.clone(),
        last_auth: last_auth.clone(),
    };

    async fn handler(
        State(state): State<AppState>,
        headers: axum::http::HeaderMap,
    ) -> (axum::http::StatusCode, Json<serde_json::Value>) {
        let call = state.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(auth) = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
        {
            let mut guard = state.last_auth.lock().await;
            *guard = Some(auth.to_string());
        }

        if call == 0 {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "fail"})),
            )
        } else {
            (
                axum::http::StatusCode::OK,
                Json(serde_json::json!({
                    "model": "resp-model",
                    "choices": [{"message": {"content": "YES"}}],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 0}
                    }
                })),
            )
        }
    }

    let app = Router::new()
        .route("/v1/chat/completions", post(handler))
        .with_state(state);
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    std::env::set_var("JUDGE_API_KEY", "test-judge-key");
    let judge = LlmJudgeEvaluator::new(
        "gpt-4o-mini".into(),
        Some(format!("http://{}", addr)),
        None,
        1,
        0,
    );
    let res = judge.evaluate(&make_case("hello"), &ro("hello")).await;
    handle.abort();
    std::env::remove_var("JUDGE_API_KEY");

    assert!(res.passed, "should succeed after retry");
    assert_eq!(calls.load(Ordering::SeqCst), 2, "should retry once");
    let auth = last_auth.lock().await.clone();
    assert_eq!(
        auth.as_deref(),
        Some("Bearer test-judge-key"),
        "should forward API key"
    );
}
