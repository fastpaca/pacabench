//! Metrics aggregation utilities.
//!
//! Aggregates individual case results into summary metrics including:
//! - Accuracy, precision, recall
//! - Duration percentiles (p50, p95)
//! - LLM latency statistics
//! - Token counts
//! - Retry attempt statistics

use crate::types::{AggregatedMetrics, CaseResult};
use statrs::statistics::{Data, OrderStatistics};

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut d = Data::new(data.to_vec());
    d.percentile((p * 100.0) as usize)
}

/// Aggregate individual case results into summary metrics.
#[must_use]
pub fn aggregate_results(results: &[CaseResult]) -> AggregatedMetrics {
    if results.is_empty() {
        return AggregatedMetrics::default();
    }

    let total = results.len() as f64;
    let passed = results.iter().filter(|r| r.passed).count() as f64;
    let accuracy = passed / total;

    // Attempt statistics
    let mut total_attempts = 0u64;
    let mut max_attempts = 0u32;
    for r in results {
        total_attempts += r.attempt as u64;
        max_attempts = max_attempts.max(r.attempt);
    }
    let avg_attempts = total_attempts as f64 / total;

    // Duration percentiles
    let durations: Vec<f64> = results.iter().map(|r| r.runner_duration_ms).collect();
    let p50_duration_ms = percentile(&durations, 0.5);
    let p95_duration_ms = percentile(&durations, 0.95);

    // LLM metrics - now typed, no HashMap lookups
    let mut latencies = Vec::new();
    let mut total_input = 0u64;
    let mut total_output = 0u64;
    let mut total_cached = 0u64;
    let mut total_calls = 0u64;
    let mut total_judge_input_tokens = 0u64;
    let mut total_judge_output_tokens = 0u64;

    for r in results {
        latencies.extend(&r.llm_metrics.latency_ms);
        total_calls += r.llm_metrics.call_count;
        total_input += r.llm_metrics.input_tokens;
        total_output += r.llm_metrics.output_tokens;
        total_cached += r.llm_metrics.cached_tokens;

        if let Some(ref jm) = r.judge_metrics {
            total_judge_input_tokens += jm.input_tokens;
            total_judge_output_tokens += jm.output_tokens;
        }
    }

    let avg_llm_latency_ms = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let p50_llm_latency_ms = percentile(&latencies, 0.5);
    let p95_llm_latency_ms = percentile(&latencies, 0.95);

    AggregatedMetrics {
        accuracy,
        total_cases: results.len() as u64,
        failed_cases: (results.len() as u64).saturating_sub(passed as u64),
        p50_duration_ms,
        p95_duration_ms,
        avg_llm_latency_ms,
        p50_llm_latency_ms,
        p95_llm_latency_ms,
        total_llm_calls: total_calls,
        total_input_tokens: total_input,
        total_output_tokens: total_output,
        total_cached_tokens: total_cached,
        total_judge_input_tokens,
        total_judge_output_tokens,
        total_attempts,
        avg_attempts,
        max_attempts,
    }
}
