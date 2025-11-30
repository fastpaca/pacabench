//! Tests for the evaluators module.

use axum::{extract::State, routing::post, Json, Router};
use pacabench_core::config::EvaluatorConfig;
use pacabench_core::evaluators::{
    Evaluator, ExactMatchEvaluator, F1Evaluator, LlmJudgeEvaluator, MultipleChoiceEvaluator,
};
use pacabench_core::types::{Case, ErrorType, RunnerOutput};
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
fn f1_evaluator_handles_punctuation() {
    let ev = F1Evaluator::new(0.5);
    let res = ev.evaluate(&make_case("Hello world"), &ro("Hello, world!"));
    assert!(res.passed, "punctuation should not break token match");
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
    let ev = MultipleChoiceEvaluator::new(&EvaluatorConfig {
        r#type: "multiple_choice".into(),
        model: None,
        extra_config: Default::default(),
        additional: Default::default(),
    });
    let res = ev.evaluate(&case, &ro("A"));
    assert!(res.passed);
}

#[test]
fn multiple_choice_fallbacks_to_f1_without_choices() {
    let mut case = make_case("foo bar");
    case.metadata = HashMap::new();
    let ev = MultipleChoiceEvaluator::new(&EvaluatorConfig {
        r#type: "multiple_choice".into(),
        model: None,
        extra_config: Default::default(),
        additional: Default::default(),
    });
    let res = ev.evaluate(&case, &ro("foo bar"));
    assert!(res.passed, "fallback F1 should be used when no choices");
}

#[tokio::test(flavor = "multi_thread")]
async fn llm_judge_records_tokens_and_cost() {
    // Stub server to avoid real OpenAI calls.
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
    let res = judge.evaluate(&make_case("hello"), &ro("hello"));
    handle.abort();
    std::env::remove_var("OPENAI_API_KEY");
    std::env::remove_var("JUDGE_BASE_URL");
    assert!(res.passed);
    assert!(res.cost_usd.unwrap_or(0.0) >= 0.0);
    assert_eq!(
        res.metrics.get("input_tokens").and_then(|v| v.as_u64()),
        Some(10)
    );
    assert_eq!(
        res.metrics.get("output_tokens").and_then(|v| v.as_u64()),
        Some(2)
    );
    assert!(
        res.metrics.contains_key("latency_ms"),
        "latency should be recorded"
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
    let res = judge.evaluate(&make_case("hello"), &ro("hello"));
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
