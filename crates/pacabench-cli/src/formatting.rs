//! Output formatting for run results (show, export commands).

use crate::pricing::calculate_cost_from_metrics;
use pacabench_core::metrics::aggregate_results;
use pacabench_core::persistence::{ErrorEntry, RunMetadata, RunSummary};
use pacabench_core::types::ErrorType;
use pacabench_core::CaseResult;
use serde::Serialize;
use std::collections::BTreeMap;

// ============================================================================
// Show command formatting
// ============================================================================

pub fn print_run_list(runs: &[RunSummary], limit: usize) {
    if runs.is_empty() {
        println!("No runs found.");
        return;
    }

    println!("Runs (showing up to {limit}):");
    println!(
        "{:<28} {:<10} {:>14} {:>10}",
        "run_id", "status", "cases", "progress"
    );

    for r in runs.iter().take(limit.min(runs.len())) {
        let progress = format!("{:.0}%", r.progress * 100.0);
        let cases = if r.total_cases > 0 {
            format!("{}/{}", r.completed_cases, r.total_cases)
        } else {
            "-".into()
        };
        let status = format!("{:?}", r.status).to_lowercase();

        println!(
            "{:<28} {:<10} {:>14} {:>10}",
            r.run_id, status, cases, progress
        );
    }

    if runs.len() > limit {
        println!("... and {} more", runs.len() - limit);
    }
}

pub fn print_run_details(
    run_id: &str,
    metadata: Option<RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) {
    let status = metadata
        .as_ref()
        .map(|m| format!("{:?}", m.status).to_lowercase())
        .unwrap_or_else(|| "unknown".to_string());
    let total_cases = metadata
        .as_ref()
        .map(|m| m.total_cases)
        .unwrap_or(results.len() as u64);
    let completed_cases = metadata
        .as_ref()
        .map(|m| m.completed_cases)
        .unwrap_or(results.len() as u64);

    println!(
        "Run {run_id} [{status}] cases {completed}/{total}",
        completed = completed_cases,
        total = total_cases
    );

    if results.is_empty() {
        println!("No results yet.");
        return;
    }

    let agg = aggregate_results(results);
    let total_cost = calculate_cost_from_metrics(
        agg.total_input_tokens,
        agg.total_output_tokens,
        agg.total_cached_tokens,
    );
    let judge_cost = calculate_cost_from_metrics(
        agg.total_judge_input_tokens,
        agg.total_judge_output_tokens,
        0,
    );
    println!(
        "Accuracy {acc:.1}% | Failed {failed}",
        acc = agg.accuracy * 100.0,
        failed = agg.failed_cases
    );
    println!(
        "Duration p50={:.0}ms p95={:.0}ms | LLM latency avg/p50/p95 = {:.0}/{:.0}/{:.0} ms",
        agg.p50_duration_ms,
        agg.p95_duration_ms,
        agg.avg_llm_latency_ms,
        agg.p50_llm_latency_ms,
        agg.p95_llm_latency_ms
    );
    println!(
        "Tokens in/out: {}/{} (judge {}/{}) | LLM calls {} | Cost ${:.4} (judge ${:.4}) | Attempts avg/max {:.1}/{}",
        agg.total_input_tokens,
        agg.total_output_tokens,
        agg.total_judge_input_tokens,
        agg.total_judge_output_tokens,
        agg.total_llm_calls,
        total_cost,
        judge_cost,
        agg.avg_attempts,
        agg.max_attempts
    );
    if !errors.is_empty() {
        println!("System errors: {}", errors.len());
    }

    let mut grouped: BTreeMap<(String, String), Vec<CaseResult>> = BTreeMap::new();
    for r in results {
        grouped
            .entry((r.agent_name.clone(), r.dataset_name.clone()))
            .or_default()
            .push(r.clone());
    }

    if !grouped.is_empty() {
        println!("\nBy Agent/Dataset:");
        for ((agent, dataset), group) in grouped {
            let metrics = aggregate_results(&group);
            let passed = group.iter().filter(|r| r.passed).count();
            let total = group.len();
            let group_cost = calculate_cost_from_metrics(
                metrics.total_input_tokens,
                metrics.total_output_tokens,
                metrics.total_cached_tokens,
            );
            let group_judge_cost = calculate_cost_from_metrics(
                metrics.total_judge_input_tokens,
                metrics.total_judge_output_tokens,
                0,
            );
            let cost = group_cost + group_judge_cost;
            println!(
                "  {agent} on {dataset}: {passed}/{total} ({acc:.1}%) p50={p50:.0}ms cost=${cost:.4}",
                acc = metrics.accuracy * 100.0,
                p50 = metrics.p50_duration_ms,
            );
        }
    }

    let failures = collect_failures(results, errors);
    if !failures.is_empty() {
        println!("\nFailures (showing up to 10):");
        for line in failures.iter().take(10) {
            println!("  - {line}");
        }
        if failures.len() > 10 {
            println!("  ... and {} more", failures.len() - 10);
        }
    }
}

pub fn print_cases(
    run_id: &str,
    results: &[CaseResult],
    errors: &[ErrorEntry],
    failures_only: bool,
    limit: usize,
) {
    println!("\nCases for {run_id}:");
    let mut rows: Vec<(String, String, String, String, String)> = Vec::new();

    for r in results {
        let status = if matches!(
            r.error_type,
            ErrorType::SystemFailure | ErrorType::FatalError
        ) {
            "error"
        } else if r.passed {
            "pass"
        } else {
            "fail"
        };
        if failures_only && status == "pass" {
            continue;
        }
        let summary = r
            .output
            .clone()
            .or_else(|| r.error.clone())
            .unwrap_or_else(|| "-".into());
        rows.push((
            r.case_id.clone(),
            r.agent_name.clone(),
            r.dataset_name.clone(),
            status.to_string(),
            summary,
        ));
    }

    for e in errors {
        let status = match e.error_type {
            ErrorType::FatalError => "fatal",
            _ => "error",
        };
        let case_id = e.case_id.clone().unwrap_or_else(|| "-".into());
        let agent = e.agent_name.clone().unwrap_or_else(|| "-".into());
        let dataset = e.dataset_name.clone().unwrap_or_else(|| "-".into());
        let summary = e.error.clone().unwrap_or_else(|| "unknown error".into());
        rows.push((case_id, agent, dataset, status.into(), summary));
    }

    if rows.is_empty() {
        println!("No cases recorded.");
        return;
    }

    rows.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    println!(
        "{:<12} {:<16} {:<16} {:<8} output/error",
        "case_id", "agent", "dataset", "status"
    );
    let total = rows.len();
    for (case_id, agent, dataset, status, summary) in rows.into_iter().take(limit) {
        let truncated = if summary.len() > 80 {
            format!("{}...", &summary[..77])
        } else {
            summary
        };
        println!(
            "{:<12} {:<16} {:<16} {:<8} {}",
            case_id, agent, dataset, status, truncated
        );
    }

    if total > limit {
        println!("... showing {limit} of {total}");
    }
}

fn collect_failures(results: &[CaseResult], errors: &[ErrorEntry]) -> Vec<String> {
    let mut failures = Vec::new();
    for r in results {
        if r.passed {
            continue;
        }
        let reason = r
            .error
            .clone()
            .or_else(|| r.judge_reason.clone())
            .unwrap_or_else(|| "failed evaluation".into());
        failures.push(format!(
            "{}/{} {}: {}",
            r.dataset_name, r.case_id, r.agent_name, reason
        ));
    }
    for e in errors {
        let cid = e.case_id.clone().unwrap_or_else(|| "-".into());
        let ds = e.dataset_name.clone().unwrap_or_else(|| "-".into());
        let agent = e.agent_name.clone().unwrap_or_else(|| "-".into());
        let reason = e.error.clone().unwrap_or_else(|| "system error".into());
        failures.push(format!("{ds}/{cid} {agent}: {reason}"));
    }
    failures
}

// ============================================================================
// Export command formatting
// ============================================================================

#[derive(Debug, Serialize)]
pub struct ExportAgent {
    pub metrics: pacabench_core::types::AggregatedMetrics,
    pub results: Vec<CaseResult>,
}

#[derive(Debug, Serialize)]
pub struct ExportRun {
    pub run_id: String,
    pub status: String,
    pub start_time: Option<String>,
    pub completed_time: Option<String>,
    pub total_cases: u64,
    pub completed_cases: u64,
    pub agents: BTreeMap<String, ExportAgent>,
    pub system_errors: Vec<ErrorEntry>,
}

pub fn build_export_json(
    run_id: &str,
    metadata: Option<&RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) -> serde_json::Value {
    let mut agents = BTreeMap::new();
    for (agent, agent_results) in group_results_by_agent(results) {
        let metrics = aggregate_results(&agent_results);
        agents.insert(
            agent,
            ExportAgent {
                metrics,
                results: agent_results,
            },
        );
    }

    let export = ExportRun {
        run_id: run_id.to_string(),
        status: metadata
            .map(|m| format!("{:?}", m.status).to_lowercase())
            .unwrap_or_else(|| "unknown".to_string()),
        start_time: metadata.and_then(|m| m.start_time.clone()),
        completed_time: metadata.and_then(|m| m.completed_time.clone()),
        total_cases: metadata
            .map(|m| m.total_cases)
            .unwrap_or(results.len() as u64),
        completed_cases: metadata
            .map(|m| m.completed_cases)
            .unwrap_or(results.len() as u64),
        agents,
        system_errors: errors.to_vec(),
    };

    serde_json::to_value(export).expect("export serialization should succeed")
}

pub fn build_export_markdown(
    run_id: &str,
    metadata: Option<&RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) -> String {
    let mut md = String::new();
    let agg = aggregate_results(results);
    md.push_str(&format!("# Run: {run_id}\n\n"));
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- **Status**: {}\n",
        metadata
            .map(|m| format!("{:?}", m.status).to_lowercase())
            .unwrap_or_else(|| "unknown".to_string())
    ));
    md.push_str(&format!(
        "- **Cases**: {} / {}\n",
        metadata
            .map(|m| m.completed_cases)
            .unwrap_or(results.len() as u64),
        metadata
            .map(|m| m.total_cases)
            .unwrap_or(results.len() as u64)
    ));
    md.push_str(&format!("- **Accuracy**: {:.1}%\n", agg.accuracy * 100.0));
    md.push_str(&format!(
        "- **Duration (p50/p95)**: {:.0}ms / {:.0}ms\n",
        agg.p50_duration_ms, agg.p95_duration_ms
    ));
    md.push_str(&format!(
        "- **LLM Latency (avg/p50/p95)**: {:.0}ms / {:.0}ms / {:.0}ms\n",
        agg.avg_llm_latency_ms, agg.p50_llm_latency_ms, agg.p95_llm_latency_ms
    ));
    md.push_str(&format!(
        "- **Tokens (in/out)**: {} / {}\n",
        agg.total_input_tokens, agg.total_output_tokens
    ));
    md.push_str(&format!(
        "- **Judge Tokens (in/out)**: {} / {}\n",
        agg.total_judge_input_tokens, agg.total_judge_output_tokens
    ));
    let md_cost = calculate_cost_from_metrics(
        agg.total_input_tokens,
        agg.total_output_tokens,
        agg.total_cached_tokens,
    );
    let md_judge_cost = calculate_cost_from_metrics(
        agg.total_judge_input_tokens,
        agg.total_judge_output_tokens,
        0,
    );
    md.push_str(&format!(
        "- **Cost**: ${:.4} (judge ${:.4})\n",
        md_cost, md_judge_cost
    ));
    md.push_str(&format!(
        "- **Attempts (avg/max)**: {:.1} / {}\n",
        agg.avg_attempts, agg.max_attempts
    ));

    if let Some(grouped) = grouped_by_agent_and_dataset(results) {
        md.push_str("\n## Agent/Dataset\n\n");
        md.push_str("| Agent | Dataset | Passed/Total | Accuracy | Cost |\n");
        md.push_str("|-------|---------|--------------|----------|------|\n");
        for ((agent, dataset), group) in grouped {
            let metrics = aggregate_results(&group);
            let passed = group.iter().filter(|r| r.passed).count();
            let total = group.len();
            let row_cost = calculate_cost_from_metrics(
                metrics.total_input_tokens,
                metrics.total_output_tokens,
                metrics.total_cached_tokens,
            );
            let row_judge_cost = calculate_cost_from_metrics(
                metrics.total_judge_input_tokens,
                metrics.total_judge_output_tokens,
                0,
            );
            let cost = row_cost + row_judge_cost;
            md.push_str(&format!(
                "| {} | {} | {}/{} | {:.1}% | ${:.4} |\n",
                agent,
                dataset,
                passed,
                total,
                metrics.accuracy * 100.0,
                cost
            ));
        }
    }

    let failures = collect_failures(results, errors);
    if !failures.is_empty() {
        md.push_str("\n## Failures\n\n");
        for line in failures {
            md.push_str(&format!("- {line}\n"));
        }
    }

    md
}

fn group_results_by_agent(results: &[CaseResult]) -> BTreeMap<String, Vec<CaseResult>> {
    let mut grouped: BTreeMap<String, Vec<CaseResult>> = BTreeMap::new();
    for r in results {
        grouped
            .entry(r.agent_name.clone())
            .or_default()
            .push(r.clone());
    }
    grouped
}

fn grouped_by_agent_and_dataset(
    results: &[CaseResult],
) -> Option<BTreeMap<(String, String), Vec<CaseResult>>> {
    if results.is_empty() {
        return None;
    }
    let mut grouped: BTreeMap<(String, String), Vec<CaseResult>> = BTreeMap::new();
    for r in results {
        grouped
            .entry((r.agent_name.clone(), r.dataset_name.clone()))
            .or_default()
            .push(r.clone());
    }
    Some(grouped)
}
