//! Evaluator implementations (exact match, F1, multiple choice, LLM judge).

use crate::config::{EvaluatorConfig, MultipleChoiceFallback};
use crate::runner::RunnerOutput;
use crate::types::{Case, EvaluationResult, JudgeMetrics};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashSet;
use std::env;
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, get_bpe_from_model};
use tokio::time::{sleep, Duration};

/// Trait for evaluating runner output against expected results.
///
/// Evaluators determine whether a case passed or failed based on the
/// runner's output and the expected answer from the dataset.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Evaluate the runner output against the expected result.
    async fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult;
    /// Return the evaluator type name (e.g., "exact_match", "f1", "llm_judge").
    fn kind(&self) -> &str;
}

/// Evaluator that requires exact string match between output and expected.
pub struct ExactMatchEvaluator;

#[async_trait]
impl Evaluator for ExactMatchEvaluator {
    async fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if !output.is_success() {
            return EvaluationResult::fail(0.0, "No output or error");
        }

        let expected = case.expected.as_deref().unwrap_or_default().trim();
        if expected.is_empty() {
            return EvaluationResult::pass(1.0);
        }

        let pred = output.output.as_deref().unwrap_or_default().trim();
        if pred == expected {
            EvaluationResult::pass(1.0)
        } else {
            EvaluationResult::fail(0.0, "Mismatch")
        }
    }

    fn kind(&self) -> &str {
        "exact_match"
    }
}

/// Evaluator using token-level F1 score with a configurable threshold.
///
/// Tokenizes both the output and expected using BPE (cl100k_base or GPT-2),
/// then computes precision, recall, and F1 score.
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

#[async_trait]
impl Evaluator for F1Evaluator {
    async fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if !output.is_success() {
            return EvaluationResult::fail(0.0, "No output or error");
        }

        let expected = case.expected.as_deref().unwrap_or_default().trim();
        if expected.is_empty() {
            return EvaluationResult::pass(1.0);
        }

        let pred_tokens = self.tokenize(output.output.as_deref().unwrap_or_default());
        let ref_tokens = self.tokenize(expected);

        if ref_tokens.is_empty() {
            return EvaluationResult::pass(1.0);
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

        let mut result = if f1 >= self.threshold {
            EvaluationResult::pass(f1)
        } else {
            EvaluationResult::fail(f1, format!("F1 {f1:.2} < threshold {:.2}", self.threshold))
        };
        result.f1_score = Some(f1);
        result
    }

    fn kind(&self) -> &str {
        "f1"
    }
}

/// Evaluator for multiple-choice questions.
///
/// Attempts to extract a choice letter (A, B, C, D) from the output and
/// compare it to the expected answer. Falls back to F1 or LLM judge if
/// no valid choice is extracted.
pub struct MultipleChoiceEvaluator {
    fallback: MultipleChoiceFallback,
    judge_model: String,
    f1_threshold: f64,
}

impl MultipleChoiceEvaluator {
    pub fn new(fallback: MultipleChoiceFallback, threshold: f64, model: String) -> Self {
        Self {
            fallback,
            judge_model: model,
            f1_threshold: threshold,
        }
    }

    fn extract_choice_letter(&self, text: &str, valid: &HashSet<char>) -> Option<char> {
        use once_cell::sync::Lazy;
        static LETTERS_RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"[A-Za-z]").expect("valid regex"));

        for cap in LETTERS_RE.find_iter(text) {
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
        use once_cell::sync::Lazy;
        static NON_ALPHANUM_RE: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"[^a-z0-9]+").expect("valid regex"));

        NON_ALPHANUM_RE
            .replace_all(&text.to_lowercase(), " ")
            .trim()
            .to_string()
    }
}

#[async_trait]
impl Evaluator for MultipleChoiceEvaluator {
    async fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        if !output.is_success() {
            return EvaluationResult::fail(0.0, "No output or error");
        }

        let expected = case.expected.as_deref().unwrap_or_default();
        if expected.is_empty() {
            return EvaluationResult::pass(1.0);
        }

        let expected_letter = expected
            .trim()
            .chars()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();

        let mut valid_letters = HashSet::new();
        if let Some(choices) = case.metadata.get("choices") {
            if let Some(obj) = choices.as_object() {
                for k in obj.keys() {
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
            return if passed {
                EvaluationResult::pass(1.0)
            } else {
                EvaluationResult::fail(0.0, format!("pred={letter}, expected={expected_letter}"))
            };
        }

        if !valid_letters.is_empty() {
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
                    return if passed {
                        EvaluationResult::pass(1.0)
                    } else {
                        EvaluationResult::fail(
                            0.0,
                            format!("matched={key_letter}, expected={expected_letter}"),
                        )
                    };
                }
            }
        }

        match self.fallback {
            MultipleChoiceFallback::F1 => {
                F1Evaluator::new(self.f1_threshold)
                    .evaluate(case, output)
                    .await
            }
            MultipleChoiceFallback::Judge => {
                LlmJudgeEvaluator::new(self.judge_model.clone(), None, None, 2, 200)
                    .evaluate(case, output)
                    .await
            }
            MultipleChoiceFallback::None => {
                EvaluationResult::fail(0.0, "No valid choice extracted")
            }
        }
    }

    fn kind(&self) -> &str {
        "multiple_choice"
    }
}

/// Evaluator that uses an LLM to judge semantic equivalence.
///
/// Prompts the model to determine if the output is semantically equivalent
/// to the expected answer, returning YES or NO.
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
        if !output.is_success() {
            return Err(anyhow!("No output or error"));
        }

        let api_key = self
            .api_key
            .clone()
            .or_else(|| env::var("JUDGE_API_KEY").ok())
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| anyhow!("Missing JUDGE_API_KEY/OPENAI_API_KEY"))?;

        let prompt = format!(
            "You are evaluating if a model's answer is semantically equivalent to the expected answer.\n\n\
             Question: {}\n\n\
             Expected Answer: {}\n\n\
             Model's Answer: {}\n\n\
             Respond with ONLY YES or NO.",
            case.input,
            case.expected.as_deref().unwrap_or_default(),
            output.output.as_deref().unwrap_or_default()
        );

        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        });

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url
                .clone()
                .unwrap_or_else(|| crate::utils::DEFAULT_OPENAI_BASE_URL.into())
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

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let json: serde_json::Value = resp.json().await.unwrap_or_default();

        let content = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_uppercase();

        let passed = content.starts_with("YES");

        let usage = json.get("usage");
        let input_tokens = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached_tokens = usage
            .and_then(|u| u.get("prompt_tokens_details"))
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        Ok(EvaluationResult {
            passed,
            score: if passed { 1.0 } else { 0.0 },
            reason: Some(content),
            latency_ms,
            f1_score: None,
            judge_metrics: Some(JudgeMetrics {
                input_tokens,
                output_tokens,
                cached_tokens,
                latency_ms,
                model: Some(self.model.clone()),
            }),
        })
    }
}

#[async_trait]
impl Evaluator for LlmJudgeEvaluator {
    async fn evaluate(&self, case: &Case, output: &RunnerOutput) -> EvaluationResult {
        for attempt in 0..=self.max_retries {
            match self.evaluate_once(case, output).await {
                Ok(res) => return res,
                Err(err) => {
                    if attempt >= self.max_retries {
                        return EvaluationResult::fail(0.0, format!("Judge error: {err}"));
                    }
                    let backoff = self.backoff_ms * (attempt as u64 + 1);
                    sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
        EvaluationResult::fail(0.0, "Judge error: exhausted retries")
    }

    fn kind(&self) -> &str {
        "llm_judge"
    }
}

/// Create an evaluator from configuration.
pub fn get_evaluator(cfg: &EvaluatorConfig) -> Result<Arc<dyn Evaluator>> {
    match cfg {
        EvaluatorConfig::ExactMatch => Ok(Arc::new(ExactMatchEvaluator)),
        EvaluatorConfig::F1 { threshold } => Ok(Arc::new(F1Evaluator::new(*threshold))),
        EvaluatorConfig::MultipleChoice {
            fallback,
            threshold,
            model,
        } => Ok(Arc::new(MultipleChoiceEvaluator::new(
            *fallback,
            *threshold,
            model.clone(),
        ))),
        EvaluatorConfig::LlmJudge {
            model,
            base_url,
            api_key,
            max_retries,
            backoff_ms,
        } => Ok(Arc::new(LlmJudgeEvaluator::new(
            model.clone(),
            base_url.clone(),
            api_key.clone(),
            *max_retries,
            *backoff_ms,
        ))),
    }
}
