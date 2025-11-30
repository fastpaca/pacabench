//! Shared data types for PacaBench.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

type JsonMap = HashMap<String, Value>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Case {
    pub case_id: String,
    pub dataset_name: String,
    pub input: String,
    #[serde(default)]
    pub expected: Option<String>,
    #[serde(default)]
    pub history: Vec<Value>,
    #[serde(default)]
    pub metadata: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunnerMetrics {
    #[serde(default)]
    pub call_count: Option<i64>,
    #[serde(default)]
    pub input_tokens: Option<i64>,
    #[serde(default)]
    pub output_tokens: Option<i64>,
    #[serde(default)]
    pub cache_read_tokens: Option<i64>,
    #[serde(default)]
    pub cache_write_tokens: Option<i64>,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub latency_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    #[default]
    None,
    TaskFailure,
    SystemFailure,
    FatalError,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunnerOutput {
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub metrics: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default)]
    pub error_type: ErrorType,
    #[serde(default)]
    pub error_traceback: Option<String>,
    #[serde(default)]
    pub retry_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationResult {
    pub passed: bool,
    pub score: f64,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub evaluator_latency_ms: f64,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub metrics: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub dataset_name: String,
    pub agent_name: String,
    pub passed: bool,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_type: ErrorType,
    #[serde(default)]
    pub runner_duration_ms: f64,
    #[serde(default)]
    pub llm_metrics: JsonMap,
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub f1_score: Option<f64>,
    #[serde(default)]
    pub f1_passed: Option<bool>,
    #[serde(default)]
    pub judge_passed: Option<bool>,
    #[serde(default)]
    pub judge_reason: Option<String>,
    #[serde(default)]
    pub judge_metrics: JsonMap,
    #[serde(default)]
    pub judge_cost_usd: Option<f64>,
    #[serde(default)]
    pub extra: JsonMap,
}

fn default_attempt() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AggregatedMetrics {
    #[serde(default)]
    pub accuracy: f64,
    #[serde(default)]
    pub precision: f64,
    #[serde(default)]
    pub total_cases: u64,
    #[serde(default)]
    pub failed_cases: u64,
    #[serde(default)]
    pub p50_duration_ms: f64,
    #[serde(default)]
    pub p95_duration_ms: f64,
    #[serde(default)]
    pub avg_llm_latency_ms: f64,
    #[serde(default)]
    pub p50_llm_latency_ms: f64,
    #[serde(default)]
    pub p95_llm_latency_ms: f64,
    #[serde(default)]
    pub total_llm_calls: u64,
    #[serde(default)]
    pub total_input_tokens: u64,
    #[serde(default)]
    pub total_output_tokens: u64,
    #[serde(default)]
    pub total_cost_usd: f64,
    #[serde(default)]
    pub total_judge_cost_usd: f64,
    #[serde(default)]
    pub total_judge_input_tokens: u64,
    #[serde(default)]
    pub total_judge_output_tokens: u64,
}
