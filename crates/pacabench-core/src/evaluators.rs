//! Simple evaluator implementations (exact match, F1, multiple choice, LLM judge).

use crate::config::EvaluatorConfig;
use crate::pricing::calculate_cost;
use crate::types::{Case, EvaluationResult, RunnerOutput};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::Arc;
use tokio::runtime::Handle;

pub trait Evaluator: Send + Sync {
    fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult;
    fn kind(&self) -> &str;
}

pub struct ExactMatchEvaluator;

impl Evaluator for ExactMatchEvaluator {
    fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if output.error.is_some() || output.output.is_none() {
            return EvaluationResult {
                passed: false,
                score: 0.0,
                reason: Some("No output or error".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let expected = case.expected.as_deref().unwrap_or_default().trim();
        if expected.is_empty() {
            return EvaluationResult {
                passed: true,
                score: 1.0,
                reason: Some("No expected output provided".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let pred = output.output.as_deref().unwrap_or_default().trim();
        let passed = pred == expected;
        EvaluationResult {
            passed,
            score: if passed { 1.0 } else { 0.0 },
            reason: Some(if passed { "Exact match" } else { "Mismatch" }.into()),
            evaluator_latency_ms: 0.0,
            cost_usd: None,
            metrics: HashMap::new(),
        }
    }

    fn kind(&self) -> &str {
        "exact_match"
    }
}

pub struct F1Evaluator {
    threshold: f64,
}

impl F1Evaluator {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Evaluator for F1Evaluator {
    fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if output.error.is_some() || output.output.is_none() {
            return EvaluationResult {
                passed: false,
                score: 0.0,
                reason: Some("No output or error".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let expected = case.expected.as_deref().unwrap_or_default().trim();
        if expected.is_empty() {
            return EvaluationResult {
                passed: true,
                score: 1.0,
                reason: Some("No expected output provided".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let pred_text = output.output.as_deref().unwrap_or_default().to_lowercase();
        let pred_tokens: HashSet<_> = pred_text.split_whitespace().collect();

        let expected_lower = expected.to_lowercase();
        let ref_tokens: HashSet<_> = expected_lower.split_whitespace().collect();

        if ref_tokens.is_empty() {
            return EvaluationResult {
                passed: true,
                score: 1.0,
                reason: Some("Empty reference".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }

        let common = pred_tokens.intersection(&ref_tokens).count() as f64;
        let precision = if pred_tokens.is_empty() {
            0.0
        } else {
            common / pred_tokens.len() as f64
        };
        let recall = common / ref_tokens.len() as f64;
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        };
        let passed = f1 >= self.threshold;
        EvaluationResult {
            passed,
            score: f1,
            reason: Some(format!("F1 Score: {f1:.2}")),
            evaluator_latency_ms: 0.0,
            cost_usd: None,
            metrics: HashMap::new(),
        }
    }

    fn kind(&self) -> &str {
        "f1"
    }
}

pub struct MultipleChoiceEvaluator;

impl Evaluator for MultipleChoiceEvaluator {
    fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if output.error.is_some() || output.output.is_none() {
            return EvaluationResult {
                passed: false,
                score: 0.0,
                reason: Some("No output or error".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let expected = case.expected.as_deref().unwrap_or_default();
        if expected.is_empty() {
            return EvaluationResult {
                passed: true,
                score: 1.0,
                reason: Some("No expected output provided".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let expected_letter = expected
            .trim()
            .chars()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();

        // Extract choices from metadata if present.
        let mut valid_letters = HashSet::new();
        if let Some(choices) = case.metadata.get("choices") {
            if let Some(obj) = choices.as_object() {
                for (k, _) in obj {
                    if let Some(c) = k.chars().next() {
                        valid_letters.insert(c.to_ascii_uppercase());
                    }
                }
            }
        }

        let pred_letter = output
            .output
            .as_deref()
            .unwrap_or_default()
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or_default()
            .to_ascii_uppercase();

        let passed = if valid_letters.is_empty() {
            pred_letter == expected_letter
        } else {
            valid_letters.contains(&pred_letter) && pred_letter == expected_letter
        };

        EvaluationResult {
            passed,
            score: if passed { 1.0 } else { 0.0 },
            reason: Some(format!("pred={pred_letter}, expected={expected_letter}")),
            evaluator_latency_ms: 0.0,
            cost_usd: None,
            metrics: HashMap::new(),
        }
    }

    fn kind(&self) -> &str {
        "multiple_choice"
    }
}

pub struct LlmJudgeEvaluator {
    model: String,
    base_url: Option<String>,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl LlmJudgeEvaluator {
    pub fn new(model: String, base_url: Option<String>) -> Self {
        Self {
            model,
            base_url,
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            client: reqwest::Client::new(),
        }
    }

    async fn evaluate_async(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if output.error.is_some() || output.output.is_none() {
            return EvaluationResult {
                passed: false,
                score: 0.0,
                reason: Some("No output or error".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }
        let api_key = match &self.api_key {
            Some(k) => k.clone(),
            None => {
                return EvaluationResult {
                    passed: false,
                    score: 0.0,
                    reason: Some("Missing OPENAI_API_KEY".into()),
                    evaluator_latency_ms: 0.0,
                    cost_usd: None,
                    metrics: HashMap::new(),
                };
            }
        };

        let prompt = format!(
            "You are evaluating if a model's answer is semantically equivalent to the expected answer.\n\nQuestion: {}\n\nExpected Answer: {}\n\nModel's Answer: {}\n\nRespond with ONLY YES or NO.",
            case.input,
            case.expected.as_deref().unwrap_or_default(),
            output.output.as_deref().unwrap_or_default()
        );

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        });

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com".into())
        );

        let start = std::time::Instant::now();
        let resp = self
            .client
            .post(&url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await;

        match resp {
            Ok(r) => {
                let latency = start.elapsed().as_millis() as f64;
                let json: serde_json::Value =
                    r.json().await.unwrap_or_else(|_| serde_json::json!({}));
                let content = json
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_uppercase();
                let passed = content.starts_with("YES");

                let usage = json.get("usage").cloned().unwrap_or_default();
                let metrics = usage
                    .as_object()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<HashMap<_, _>>();

                // Calculate cost from usage tokens
                let input_tokens = usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let output_tokens = usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let cached_tokens = usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let cost = calculate_cost(&self.model, input_tokens, output_tokens, cached_tokens);

                EvaluationResult {
                    passed,
                    score: if passed { 1.0 } else { 0.0 },
                    reason: Some(content),
                    evaluator_latency_ms: latency,
                    cost_usd: if cost > 0.0 { Some(cost) } else { None },
                    metrics,
                }
            }
            Err(e) => EvaluationResult {
                passed: false,
                score: 0.0,
                reason: Some(format!("Judge error: {e}")),
                evaluator_latency_ms: start.elapsed().as_millis() as f64,
                cost_usd: None,
                metrics: HashMap::new(),
            },
        }
    }
}

impl Evaluator for LlmJudgeEvaluator {
    fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        // Try to use existing runtime, otherwise create a new one
        match Handle::try_current() {
            Ok(handle) => {
                // We're in an async context, use block_in_place to avoid nested runtime
                tokio::task::block_in_place(|| handle.block_on(self.evaluate_async(case, output)))
            }
            Err(_) => {
                // No runtime, create a temporary one
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(self.evaluate_async(case, output))
            }
        }
    }

    fn kind(&self) -> &str {
        "llm_judge"
    }
}

pub fn get_evaluator(cfg: &EvaluatorConfig) -> Result<Arc<dyn Evaluator>> {
    match cfg.r#type.as_str() {
        "exact_match" => Ok(Arc::new(ExactMatchEvaluator)),
        "f1" => {
            let threshold = cfg
                .extra_config
                .get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            Ok(Arc::new(F1Evaluator::new(threshold)))
        }
        "multiple_choice" => Ok(Arc::new(MultipleChoiceEvaluator)),
        "llm_judge" => {
            let model = cfg.model.clone().unwrap_or_else(|| "gpt-4o-mini".into());
            let base_url = env::var("OPENAI_BASE_URL").ok();
            Ok(Arc::new(LlmJudgeEvaluator::new(model, base_url)))
        }
        other => Err(anyhow!("unsupported evaluator type: {other}")),
    }
}
