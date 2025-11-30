//! Metrics aggregation utilities.
//!
//! Aggregates individual case results into summary metrics including:
//! - Accuracy, precision, recall
//! - Duration percentiles (p50, p95)
//! - LLM latency statistics
//! - Token counts and costs
//! - Retry attempt statistics

use crate::types::{AggregatedMetrics, CaseResult};

/// Calculate the percentile value from a sorted slice.
fn percentile(data: &mut [f64], pct: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (data.len() - 1) as f64 * pct;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        data[lower]
    } else {
        let w = idx - lower as f64;
        data[lower] * (1.0 - w) + data[upper] * w
    }
}

/// Aggregate individual case results into summary metrics.
///
/// Computes accuracy, latency percentiles, token counts, costs, and retry statistics.
/// Returns default metrics if `results` is empty.
#[must_use]
pub fn aggregate_results(results: &[CaseResult]) -> AggregatedMetrics {
    if results.is_empty() {
        return AggregatedMetrics::default();
    }

    let total = results.len() as f64;
    let passed = results.iter().filter(|r| r.passed).count() as f64;
    let accuracy = passed / total;

    let mut total_attempts = 0u64;
    let mut max_attempts = 0u32;
    for r in results {
        total_attempts += r.attempt as u64;
        max_attempts = max_attempts.max(r.attempt);
    }
    let avg_attempts = if total > 0.0 {
        total_attempts as f64 / total
    } else {
        0.0
    };

    let mut durations: Vec<f64> = results.iter().map(|r| r.runner_duration_ms).collect();
    let p50_duration_ms = percentile(&mut durations.clone(), 0.5);
    let p95_duration_ms = percentile(&mut durations, 0.95);

    let mut latencies = Vec::new();
    let mut total_cost = 0.0;
    let mut total_input = 0;
    let mut total_output = 0;
    let mut total_calls = 0;
    let mut total_judge_cost = 0.0;
    let mut total_judge_input_tokens = 0;
    let mut total_judge_output_tokens = 0;
    for r in results {
        if let Some(ms_list) = r.llm_metrics.get("llm_latency_ms") {
            if let Some(arr) = ms_list.as_array() {
                for v in arr {
                    if let Some(f) = v.as_f64() {
                        latencies.push(f);
                    }
                }
            }
        }
        total_calls += r
            .llm_metrics
            .get("llm_call_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        total_cost += r
            .llm_metrics
            .get("llm_total_cost_usd")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        total_input += r
            .llm_metrics
            .get("llm_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        total_output += r
            .llm_metrics
            .get("llm_output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        total_judge_cost += r.judge_cost_usd.unwrap_or(0.0);
        if let Some(jmetrics) = r.judge_metrics.get("input_tokens") {
            total_judge_input_tokens += jmetrics.as_u64().unwrap_or(0);
        }
        if let Some(jmetrics) = r.judge_metrics.get("output_tokens") {
            total_judge_output_tokens += jmetrics.as_u64().unwrap_or(0);
        }
    }

    let avg_lat = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().copied().sum::<f64>() / latencies.len() as f64
    };
    let p50_llm_latency_ms = percentile(&mut latencies.clone(), 0.5);
    let p95_llm_latency_ms = percentile(&mut latencies, 0.95);

    AggregatedMetrics {
        accuracy,
        total_cases: results.len() as u64,
        failed_cases: (results.len() as u64).saturating_sub(passed as u64),
        p50_duration_ms,
        p95_duration_ms,
        avg_llm_latency_ms: avg_lat,
        p50_llm_latency_ms,
        p95_llm_latency_ms,
        total_llm_calls: total_calls,
        total_input_tokens: total_input,
        total_output_tokens: total_output,
        total_cost_usd: total_cost,
        total_judge_cost_usd: total_judge_cost,
        total_judge_input_tokens,
        total_judge_output_tokens,
        // With only pass/fail available, precision and recall mirror accuracy.
        precision: accuracy,
        recall: accuracy,
        total_attempts,
        avg_attempts,
        max_attempts,
    }
}
