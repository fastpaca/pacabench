//! Simple evaluator implementations (exact match, F1, multiple choice, LLM judge).

use crate::config::EvaluatorConfig;
use crate::pricing::calculate_cost;
use crate::types::{Case, EvaluationResult, RunnerOutput};
use anyhow::{anyhow, Result};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, get_bpe_from_model};
use tokio::runtime::Handle;
use tokio::time::{sleep, Duration};

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

    fn tokenize(&self, text: &str) -> HashSet<String> {
        let cleaned = text.trim().to_lowercase();
        if cleaned.is_empty() {
            return HashSet::new();
        }

        if let Ok(bpe) = cl100k_base() {
            return bpe
                .encode_with_special_tokens(&cleaned)
                .into_iter()
                .map(|t| t.to_string())
                .collect();
        }

        if let Ok(bpe) = get_bpe_from_model("gpt2") {
            return bpe
                .encode_with_special_tokens(&cleaned)
                .into_iter()
                .map(|t| t.to_string())
                .collect();
        }

        cleaned
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
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
        let pred_tokens = self.tokenize(&pred_text);

        let expected_lower = expected.to_lowercase();
        let ref_tokens = self.tokenize(&expected_lower);

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

enum MultipleChoiceFallback {
    F1,
    Judge,
    None,
}

pub struct MultipleChoiceEvaluator {
    fallback: MultipleChoiceFallback,
    judge_model: String,
    f1_threshold: f64,
}

impl MultipleChoiceEvaluator {
    pub fn new(cfg: &EvaluatorConfig) -> Self {
        let fallback_str = cfg
            .extra_config
            .get("fallback")
            .and_then(|v| v.as_str())
            .unwrap_or("f1");
        let fallback = match fallback_str {
            "judge" => MultipleChoiceFallback::Judge,
            "none" => MultipleChoiceFallback::None,
            _ => MultipleChoiceFallback::F1,
        };
        let f1_threshold = cfg
            .extra_config
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let judge_model = cfg.model.clone().unwrap_or_else(|| "gpt-4o-mini".into());
        Self {
            fallback,
            judge_model,
            f1_threshold,
        }
    }

    fn extract_choice_letter(&self, text: &str, valid: &HashSet<char>) -> Option<char> {
        let letters = Regex::new(r"[A-Za-z]").unwrap();
        for cap in letters.find_iter(text) {
            let c = cap
                .as_str()
                .chars()
                .next()
                .unwrap_or_default()
                .to_ascii_uppercase();
            if valid.contains(&c) {
                return Some(c);
            }
        }
        None
    }

    fn clean_text(&self, text: &str) -> String {
        let letters = Regex::new(r"[^a-z0-9]+").unwrap();
        letters
            .replace_all(&text.to_lowercase(), " ")
            .trim()
            .to_string()
    }
}

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

        let pred_letter = self
            .extract_choice_letter(output.output.as_deref().unwrap_or_default(), &valid_letters);

        if let Some(letter) = pred_letter.filter(|_| !valid_letters.is_empty()) {
            let passed = letter == expected_letter;
            return EvaluationResult {
                passed,
                score: if passed { 1.0 } else { 0.0 },
                reason: Some(format!("pred={letter}, expected={expected_letter}")),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            };
        }

        if !valid_letters.is_empty() {
            // Try to match cleaned choice values
            let cleaned_output = self.clean_text(output.output.as_deref().unwrap_or_default());
            for (k, v) in case
                .metadata
                .get("choices")
                .and_then(|c| c.as_object())
                .into_iter()
                .flatten()
            {
                let key_letter = k.chars().next().unwrap_or_default().to_ascii_uppercase();
                let cleaned_val = self.clean_text(v.as_str().unwrap_or_default());
                if !cleaned_val.is_empty() && cleaned_val == cleaned_output {
                    let passed = key_letter == expected_letter;
                    return EvaluationResult {
                        passed,
                        score: if passed { 1.0 } else { 0.0 },
                        reason: Some(format!(
                            "matched_choice_value={key_letter}, expected={expected_letter}"
                        )),
                        evaluator_latency_ms: 0.0,
                        cost_usd: None,
                        metrics: HashMap::new(),
                    };
                }
            }
        }

        match self.fallback {
            MultipleChoiceFallback::F1 => {
                let f1 = F1Evaluator::new(self.f1_threshold);
                f1.evaluate(case, output)
            }
            MultipleChoiceFallback::Judge => {
                let judge = LlmJudgeEvaluator::new(self.judge_model.clone(), None, None, 2, 200);
                judge.evaluate(case, output)
            }
            MultipleChoiceFallback::None => EvaluationResult {
                passed: false,
                score: 0.0,
                reason: Some("No valid choice extracted".into()),
                evaluator_latency_ms: 0.0,
                cost_usd: None,
                metrics: HashMap::new(),
            },
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
    max_retries: usize,
    backoff_ms: u64,
}

impl LlmJudgeEvaluator {
    pub fn new(
        model: String,
        base_url: Option<String>,
        api_key: Option<String>,
        max_retries: usize,
        backoff_ms: u64,
    ) -> Self {
        let base_url = base_url
            .or_else(|| env::var("JUDGE_BASE_URL").ok())
            .or_else(|| env::var("OPENAI_BASE_URL").ok());
        let api_key = api_key
            .or_else(|| env::var("JUDGE_API_KEY").ok())
            .or_else(|| env::var("OPENAI_API_KEY").ok());
        Self {
            model,
            base_url,
            api_key,
            client: reqwest::Client::new(),
            max_retries,
            backoff_ms,
        }
    }

    async fn evaluate_once(&self, case: &Case, output: &RunnerOutput) -> Result<EvaluationResult> {
        if output.error.is_some() || output.output.is_none() {
            return Err(anyhow!("No output or error"));
        }
        let api_key = self
            .api_key
            .clone()
            .or_else(|| env::var("JUDGE_API_KEY").ok())
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| anyhow!("Missing JUDGE_API_KEY/OPENAI_API_KEY"))?;

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
            .await
            .map_err(|e| anyhow!("Judge request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Judge HTTP {status}: {text}"));
        }

        let latency = start.elapsed().as_millis() as f64;
        let json: serde_json::Value = resp.json().await.unwrap_or_else(|_| serde_json::json!({}));
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

        let mut metrics = HashMap::new();
        metrics.insert("input_tokens".into(), serde_json::json!(input_tokens));
        metrics.insert("output_tokens".into(), serde_json::json!(output_tokens));
        metrics.insert(
            "total_tokens".into(),
            serde_json::json!(input_tokens + output_tokens),
        );
        metrics.insert("latency_ms".into(), serde_json::json!(latency));
        metrics.insert(
            "prompt_tokens_details".into(),
            usage
                .get("prompt_tokens_details")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({ "cached_tokens": 0 })),
        );
        if cost > 0.0 {
            metrics.insert("cost_usd".into(), serde_json::json!(cost));
        }

        Ok(EvaluationResult {
            passed,
            score: if passed { 1.0 } else { 0.0 },
            reason: Some(content),
            evaluator_latency_ms: latency,
            cost_usd: if cost > 0.0 { Some(cost) } else { None },
            metrics,
        })
    }

    async fn evaluate_with_retries(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        for attempt in 0..=self.max_retries {
            match self.evaluate_once(case, output).await {
                Ok(res) => return res,
                Err(err) => {
                    if attempt >= self.max_retries {
                        return EvaluationResult {
                            passed: false,
                            score: 0.0,
                            reason: Some(format!("Judge error: {err}")),
                            evaluator_latency_ms: 0.0,
                            cost_usd: None,
                            metrics: HashMap::new(),
                        };
                    }
                    let backoff = self.backoff_ms * (attempt as u64 + 1);
                    sleep(Duration::from_millis(backoff)).await;
                }
            }
        }

        EvaluationResult {
            passed: false,
            score: 0.0,
            reason: Some("Judge error: exhausted retries".into()),
            evaluator_latency_ms: 0.0,
            cost_usd: None,
            metrics: HashMap::new(),
        }
    }
}

impl Evaluator for LlmJudgeEvaluator {
    fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        match Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(self.evaluate_with_retries(case, output))
            }),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(self.evaluate_with_retries(case, output))
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
        "multiple_choice" => Ok(Arc::new(MultipleChoiceEvaluator::new(cfg))),
        "llm_judge" => {
            let model = cfg.model.clone().unwrap_or_else(|| "gpt-4o-mini".into());
            let base_url = cfg
                .extra_config
                .get("base_url")
                .and_then(|v| v.as_str())
                .map(String::from);
            let api_key = cfg
                .extra_config
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(String::from);
            let max_retries = cfg
                .extra_config
                .get("max_retries")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let backoff_ms = cfg
                .extra_config
                .get("backoff_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(200);
            Ok(Arc::new(LlmJudgeEvaluator::new(
                model,
                base_url,
                api_key,
                max_retries,
                backoff_ms,
            )))
        }
        other => Err(anyhow!("unsupported evaluator type: {other}")),
    }
}
