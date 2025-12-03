//! CLI for PacaBench - a local-first benchmark harness for LLM agents.

mod pricing;
mod progress;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use handlebars::Handlebars;
use pacabench_core::config::ConfigOverrides;
use pacabench_core::metrics::aggregate_results;
use pacabench_core::persistence::{
    list_run_summaries, ErrorEntry, RunMetadata, RunStore, RunSummary,
};
use pacabench_core::types::ErrorType;
use pacabench_core::{Benchmark, CaseResult, Config};
use pricing::calculate_cost_from_metrics;
use progress::ProgressDisplay;
use rust_embed::Embed;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use toml_edit::DocumentMut;

#[derive(Embed)]
#[folder = "templates/init/"]
struct InitTemplates;

#[derive(Debug, Parser)]
#[command(
    name = "pacabench",
    about = "A local-first benchmark harness for LLM agents"
)]
struct Cli {
    /// Path to the benchmark configuration file.
    #[arg(short, long, default_value = "pacabench.yaml")]
    config: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Load and print the parsed configuration.
    ShowConfig,

    /// Run the benchmark.
    Run {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        runs_dir: Option<String>,
        #[arg(long)]
        cache_dir: Option<String>,
        /// Force a new run even if run_id exists with different config.
        #[arg(long)]
        force_new: bool,
        /// Comma-separated list of agents to run (e.g. --agents agent1,agent2).
        #[arg(long, short = 'a')]
        agents: Option<String>,
        /// Override concurrency (number of parallel workers).
        #[arg(long)]
        concurrency: Option<usize>,
        /// Override timeout in seconds.
        #[arg(long)]
        timeout: Option<f64>,
    },

    /// Show aggregated metrics for a run id.
    Show {
        /// Run ID (supports partial match).
        #[arg()]
        run_id: Option<String>,
        #[arg(long)]
        runs_dir: Option<String>,
        /// Show individual case rows.
        #[arg(long)]
        cases: bool,
        /// Show only failures in the case list.
        #[arg(long)]
        failures: bool,
        /// Limit number of cases shown.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },

    /// Retry failed cases from a previous run.
    Retry {
        /// Run ID to retry from.
        #[arg()]
        run_id: String,
        #[arg(long)]
        runs_dir: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Export results to JSON or Markdown.
    Export {
        /// Run ID to export.
        #[arg()]
        run_id: String,
        /// Output format: json or markdown.
        #[arg(long, default_value = "json")]
        format: String,
        /// Output file path (defaults to stdout).
        #[arg(short, long)]
        output: Option<String>,
        #[arg(long)]
        runs_dir: Option<String>,
    },

    /// Initialize a new pacabench project with example files.
    Init {
        /// Name for the benchmark. Defaults to project name from pyproject.toml if present.
        #[arg(long)]
        name: Option<String>,
        /// Overwrite existing files.
        #[arg(long)]
        force: bool,
        /// Skip updating pyproject.toml.
        #[arg(long)]
        no_pyproject: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = PathBuf::from(&cli.config);

    if let Some(Command::Init {
        name,
        force,
        no_pyproject,
    }) = cli.command
    {
        // Determine project name: explicit --name > pyproject.toml > default
        let project_name = name
            .unwrap_or_else(|| read_pyproject_name().unwrap_or_else(|| "my-benchmark".to_string()));
        init_project(&project_name, force, !no_pyproject)?;
        return Ok(());
    }

    // Build overrides based on command
    let overrides = match &cli.command {
        Some(Command::Run {
            runs_dir,
            cache_dir,
            concurrency,
            timeout,
            ..
        }) => ConfigOverrides {
            concurrency: *concurrency,
            timeout_seconds: *timeout,
            max_retries: None,
            runs_dir: runs_dir.clone(),
            cache_dir: cache_dir.clone(),
        },
        _ => ConfigOverrides::default(),
    };

    // Load config
    let mut config = Config::from_file(&config_path, overrides)?;

    match cli.command {
        Some(Command::ShowConfig) => {
            println!(
                "Loaded benchmark '{}': {} agent(s), {} dataset(s).",
                config.name,
                config.agents.len(),
                config.datasets.len()
            );
            let yaml = serde_yaml::to_string(&config)?;
            println!("{yaml}");
        }
        Some(Command::Run {
            limit,
            run_id,
            agents,
            ..
        }) => {
            // Filter agents if specified
            if let Some(agents_filter) = agents {
                let filter_set: std::collections::HashSet<&str> =
                    agents_filter.split(',').map(|s| s.trim()).collect();
                let available: Vec<String> = config.agents.iter().map(|a| a.name.clone()).collect();
                config
                    .agents
                    .retain(|a| filter_set.contains(a.name.as_str()));
                if config.agents.is_empty() {
                    return Err(anyhow!(
                        "No agents found matching filter: {}. Available: {}",
                        agents_filter,
                        available.join(", ")
                    ));
                }
                println!(
                    "Running {} agent(s): {}",
                    config.agents.len(),
                    config
                        .agents
                        .iter()
                        .map(|a| a.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            let bench = Benchmark::new(config);
            let events = bench.subscribe();

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let display = ProgressDisplay::new();
                let display_handle = tokio::spawn(display.run(events));

                let result = bench.run(run_id, limit).await;

                let _ = display_handle.await;

                result
            })?;
        }
        Some(Command::Show {
            run_id,
            runs_dir,
            cases,
            failures,
            limit,
        }) => {
            let runs_dir = runs_dir.map(PathBuf::from).unwrap_or(config.runs_dir);
            if run_id.is_none() {
                let runs = list_run_summaries(&runs_dir)?;
                print_run_list(&runs, limit);
                return Ok(());
            }

            let partial = run_id.as_ref().expect("guarded above");
            let resolved_id = resolve_run_id(&runs_dir, partial)?;
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let errors = store.load_errors()?;
            let metadata = store.read_metadata()?;
            print_run_details(&resolved_id, metadata, &results, &errors);
            if cases {
                print_cases(&resolved_id, &results, &errors, failures, limit);
            }
        }
        Some(Command::Retry {
            run_id,
            runs_dir,
            limit,
        }) => {
            if let Some(dir) = runs_dir {
                config.runs_dir = PathBuf::from(dir);
            }

            let resolved_id = resolve_run_id(&config.runs_dir, &run_id)?;

            let store = RunStore::new(config.runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let failed_count = results.iter().filter(|r| !r.passed).count();

            if failed_count == 0 {
                println!("No failed cases to retry in run {resolved_id}");
                return Ok(());
            }

            println!(
                "Retrying {} failed cases from run {resolved_id}",
                failed_count
            );

            let bench = Benchmark::new(config);
            let events = bench.subscribe();

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let display = ProgressDisplay::new();
                let display_handle = tokio::spawn(display.run(events));
                let result = bench.run(Some(resolved_id), limit).await;
                let _ = display_handle.await;
                result
            })?;
        }
        Some(Command::Export {
            run_id,
            format,
            output,
            runs_dir,
        }) => {
            let runs_dir = runs_dir.map(PathBuf::from).unwrap_or(config.runs_dir);
            let resolved_id = resolve_run_id(&runs_dir, &run_id)?;
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let errors = store.load_errors()?;
            let metadata = store.read_metadata()?;

            let content = match format.as_str() {
                "json" => serde_json::to_string_pretty(&build_export_json(
                    &resolved_id,
                    metadata.as_ref(),
                    &results,
                    &errors,
                ))?,
                "markdown" | "md" => {
                    build_export_markdown(&resolved_id, metadata.as_ref(), &results, &errors)
                }
                _ => return Err(anyhow!("unsupported format: {format}")),
            };

            match output {
                Some(path) => {
                    fs::write(&path, &content)?;
                    println!("Exported to {path}");
                }
                None => println!("{content}"),
            }
        }
        Some(Command::Init { .. }) => unreachable!(),
        None => {
            println!(
                "Loaded benchmark '{}': {} agent(s), {} dataset(s).",
                config.name,
                config.agents.len(),
                config.datasets.len()
            );
            println!("\nUse --help to see available commands.");
        }
    }

    Ok(())
}

fn print_run_list(runs: &[RunSummary], limit: usize) {
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

fn print_run_details(
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
        "Accuracy {acc:.1}% | Precision {prec:.1}% | Recall {rec:.1}% | Failed {failed}",
        acc = agg.accuracy * 100.0,
        prec = agg.precision * 100.0,
        rec = agg.recall * 100.0,
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

fn print_cases(
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

#[derive(Debug, Serialize)]
struct ExportAgent {
    metrics: pacabench_core::types::AggregatedMetrics,
    results: Vec<CaseResult>,
}

#[derive(Debug, Serialize)]
struct ExportRun {
    run_id: String,
    status: String,
    start_time: Option<String>,
    completed_time: Option<String>,
    total_cases: u64,
    completed_cases: u64,
    agents: BTreeMap<String, ExportAgent>,
    system_errors: Vec<ErrorEntry>,
}

fn build_export_json(
    run_id: &str,
    metadata: Option<&RunMetadata>,
    results: &[CaseResult],
    errors: &[ErrorEntry],
) -> serde_json::Value {
    let mut agents = BTreeMap::new();
    let mut agent_names: Vec<String> = results.iter().map(|r| r.agent_name.clone()).collect();
    agent_names.sort();
    agent_names.dedup();

    for agent in agent_names {
        let agent_results: Vec<CaseResult> = results
            .iter()
            .filter(|r| r.agent_name == agent)
            .cloned()
            .collect();
        let metrics = aggregate_results(&agent_results);
        agents.insert(
            agent.clone(),
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

fn build_export_markdown(
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
    md.push_str(&format!(
        "- **Accuracy / Precision / Recall**: {:.1}% / {:.1}% / {:.1}%\n",
        agg.accuracy * 100.0,
        agg.precision * 100.0,
        agg.recall * 100.0
    ));
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

    let mut grouped: BTreeMap<(String, String), Vec<CaseResult>> = BTreeMap::new();
    for r in results {
        grouped
            .entry((r.agent_name.clone(), r.dataset_name.clone()))
            .or_default()
            .push(r.clone());
    }

    if !grouped.is_empty() {
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

/// Read project name from pyproject.toml if it exists.
fn read_pyproject_name() -> Option<String> {
    let path = PathBuf::from("pyproject.toml");
    let content = fs::read_to_string(&path).ok()?;
    let doc: DocumentMut = content.parse().ok()?;
    doc.get("project")?
        .get("name")?
        .as_str()
        .map(|s| s.to_string())
}

fn init_project(name: &str, force: bool, update_pyproject: bool) -> Result<()> {
    let mut created_files = Vec::new();

    // Render pacabench.yaml from template
    let yaml_template = InitTemplates::get("pacabench.yaml.hbs")
        .ok_or_else(|| anyhow!("template pacabench.yaml.hbs not found"))?;
    let yaml_content = std::str::from_utf8(&yaml_template.data)?;

    let mut handlebars = Handlebars::new();
    handlebars.set_strict_mode(true);
    let rendered =
        handlebars.render_template(yaml_content, &serde_json::json!({ "name": name }))?;

    let config_path = PathBuf::from("pacabench.yaml");
    if config_path.exists() && !force {
        return Err(anyhow!(
            "pacabench.yaml already exists. Use --force to overwrite."
        ));
    }
    fs::write(&config_path, rendered)?;
    created_files.push("pacabench.yaml");

    // Create agents directory and copy qa_agent.py
    let agents_dir = PathBuf::from("agents");
    if !agents_dir.exists() {
        fs::create_dir_all(&agents_dir)?;
    }
    let agent_path = agents_dir.join("qa_agent.py");
    if agent_path.exists() && !force {
        return Err(anyhow!(
            "agents/qa_agent.py already exists. Use --force to overwrite."
        ));
    }
    let agent_template = InitTemplates::get("agents/qa_agent.py")
        .ok_or_else(|| anyhow!("template agents/qa_agent.py not found"))?;
    fs::write(&agent_path, agent_template.data.as_ref())?;
    created_files.push("agents/qa_agent.py");

    // Create data directory and copy questions.jsonl
    let data_dir = PathBuf::from("data");
    if !data_dir.exists() {
        fs::create_dir_all(&data_dir)?;
    }
    let data_path = data_dir.join("questions.jsonl");
    if data_path.exists() && !force {
        return Err(anyhow!(
            "data/questions.jsonl already exists. Use --force to overwrite."
        ));
    }
    let data_template = InitTemplates::get("data/questions.jsonl")
        .ok_or_else(|| anyhow!("template data/questions.jsonl not found"))?;
    fs::write(&data_path, data_template.data.as_ref())?;
    created_files.push("data/questions.jsonl");

    // Update .gitignore
    let gitignore_path = PathBuf::from(".gitignore");
    let gitignore_content = InitTemplates::get("gitignore.txt")
        .ok_or_else(|| anyhow!("template gitignore.txt not found"))?;
    let runs_entry = std::str::from_utf8(&gitignore_content.data)?.trim();

    if gitignore_path.exists() {
        let existing = fs::read_to_string(&gitignore_path)?;
        if !existing.contains("runs/") {
            fs::write(&gitignore_path, format!("{existing}\n{runs_entry}\n"))?;
            created_files.push(".gitignore (updated)");
        }
    } else {
        fs::write(&gitignore_path, format!("{runs_entry}\n"))?;
        created_files.push(".gitignore");
    }

    // Update pyproject.toml if requested
    if update_pyproject {
        let pyproject_path = PathBuf::from("pyproject.toml");
        if pyproject_path.exists() {
            update_pyproject_toml(&pyproject_path)?;
            created_files.push("pyproject.toml (updated)");
        } else {
            println!(
                "Note: No pyproject.toml found. Add pacabench manually:\n  \
                 uv add --dev pacabench\n  or\n  \
                 pip install pacabench"
            );
        }
    }

    // Print success message
    println!("Created pacabench project:");
    for file in &created_files {
        println!("  {file}");
    }
    println!("\nNext steps:");
    println!("  1. Set your API key: export OPENAI_API_KEY=sk-...");
    println!("  2. Install dependencies: pip install openai");
    println!("  3. Run: uv run pacabench run --limit 3");
    println!("\nSee https://github.com/fastpaca/pacabench for docs.");

    Ok(())
}

fn update_pyproject_toml(path: &PathBuf) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let mut doc: DocumentMut = content.parse()?;

    // Try to add to [dependency-groups.dev] (uv style) - it's an array directly
    if let Some(dep_groups) = doc.get_mut("dependency-groups") {
        if let Some(dep_groups_table) = dep_groups.as_table_mut() {
            if let Some(dev) = dep_groups_table.get_mut("dev") {
                if let Some(dev_array) = dev.as_array_mut() {
                    let has_pacabench = dev_array
                        .iter()
                        .any(|v| v.as_str().map(|s| s == "pacabench").unwrap_or(false));
                    if !has_pacabench {
                        dev_array.push("pacabench");
                    }
                    fs::write(path, doc.to_string())?;
                    return Ok(());
                }
            }
        }
    }

    // Fallback: try [project.optional-dependencies.dev]
    if let Some(project) = doc.get_mut("project") {
        if let Some(project_table) = project.as_table_mut() {
            if let Some(opt_deps) = project_table.get_mut("optional-dependencies") {
                if let Some(opt_deps_table) = opt_deps.as_table_mut() {
                    if let Some(dev) = opt_deps_table.get_mut("dev") {
                        if let Some(dev_array) = dev.as_array_mut() {
                            let has_pacabench = dev_array
                                .iter()
                                .any(|v| v.as_str().map(|s| s == "pacabench").unwrap_or(false));
                            if !has_pacabench {
                                dev_array.push("pacabench");
                            }
                            fs::write(path, doc.to_string())?;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    // If neither exists, add dependency-groups.dev (uv style)
    let mut dep_groups = toml_edit::Table::new();
    let mut dev_array = toml_edit::Array::new();
    dev_array.push("pacabench");
    dep_groups.insert("dev", toml_edit::value(dev_array));
    doc.insert("dependency-groups", toml_edit::Item::Table(dep_groups));
    fs::write(path, doc.to_string())?;

    Ok(())
}

fn resolve_run_id(runs_dir: &std::path::Path, partial: &str) -> Result<String> {
    let summaries = list_run_summaries(runs_dir)?;
    if summaries.is_empty() {
        return Err(anyhow!("no runs found in {}", runs_dir.display()));
    }

    if let Some(exact) = summaries.iter().find(|s| s.run_id == partial) {
        return Ok(exact.run_id.clone());
    }

    let matches: Vec<&RunSummary> = summaries
        .iter()
        .filter(|s| s.run_id.starts_with(partial) || s.run_id.contains(partial))
        .collect();

    match matches.len() {
        0 => Err(anyhow!("no run found matching '{partial}'")),
        1 => Ok(matches[0].run_id.clone()),
        _ => {
            let options: Vec<_> = matches.iter().map(|s| s.run_id.clone()).collect();
            Err(anyhow!(
                "ambiguous run ID '{partial}', matches: {}",
                options.join(", ")
            ))
        }
    }
}
