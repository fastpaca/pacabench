//! Thin CLI wrapper for the Rust rewrite of PacaBench.

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use pacabench_core::config::load_config;
use pacabench_core::metrics::aggregate_results;
use pacabench_core::orchestrator::Orchestrator;
use pacabench_core::persistence::{default_dataset_cache_dir, resolve_runs_dir, RunStore};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "pacabench", about = "Rust rewrite of the PacaBench CLI")]
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
    },

    /// Show aggregated metrics for a run id.
    Show {
        /// Run ID (supports partial match).
        #[arg()]
        run_id: String,
        #[arg(long)]
        runs_dir: Option<String>,
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

    /// Initialize a new pacabench.yaml configuration file.
    Init {
        /// Name for the benchmark.
        #[arg(long, default_value = "my-benchmark")]
        name: String,
        /// Output path.
        #[arg(short, long, default_value = "pacabench.yaml")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = PathBuf::from(&cli.config);

    if let Some(Command::Init { name, output }) = cli.command {
        let template = format!(
            r#"name: {name}
version: "0.1.0"

config:
  concurrency: 4
  timeout_seconds: 60
  max_retries: 2
  proxy:
    enabled: true

agents:
  - name: my-agent
    command: python agent.py

datasets:
  - name: my-dataset
    source: ./data

output:
  directory: ./runs
"#
        );
        fs::write(&output, template)?;
        println!("Created {output}");
        return Ok(());
    }

    let config = load_config(&config_path)?;

    let root_dir = config_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

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
            runs_dir,
            cache_dir,
            force_new,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let cache_dir = cache_dir
                .map(PathBuf::from)
                .unwrap_or_else(default_dataset_cache_dir);
            let orch = Orchestrator::new(config, root_dir, cache_dir, runs_dir);
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(orch.run(run_id, limit, force_new))?;
        }
        Some(Command::Show { run_id, runs_dir }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let resolved_id = resolve_run_id(&runs_dir, &run_id)?;
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let agg = aggregate_results(&results);
            println!(
                "Run {}: cases={}, accuracy={:.1}%, p50_dur_ms={:.0}, p95_dur_ms={:.0}",
                resolved_id,
                agg.total_cases,
                agg.accuracy * 100.0,
                agg.p50_duration_ms,
                agg.p95_duration_ms
            );
            println!(
                "LLM latency avg={:.0}ms p50={:.0}ms p95={:.0}ms tokens in={} out={} cost=${:.4}",
                agg.avg_llm_latency_ms,
                agg.p50_llm_latency_ms,
                agg.p95_llm_latency_ms,
                agg.total_input_tokens,
                agg.total_output_tokens,
                agg.total_cost_usd
            );

            // Show failure summary if any
            let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
            if !failed.is_empty() {
                println!("\nFailed cases ({}):", failed.len());
                for f in failed.iter().take(10) {
                    println!(
                        "  - {}/{}: {}",
                        f.dataset_name,
                        f.case_id,
                        f.error.as_deref().unwrap_or("evaluation failed")
                    );
                }
                if failed.len() > 10 {
                    println!("  ... and {} more", failed.len() - 10);
                }
            }
        }
        Some(Command::Retry {
            run_id,
            runs_dir,
            limit,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let resolved_id = resolve_run_id(&runs_dir, &run_id)?;

            // Load existing results and find failed cases
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
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

            let cache_dir = default_dataset_cache_dir();
            let orch = Orchestrator::new(config, root_dir, cache_dir, runs_dir);
            let rt = tokio::runtime::Runtime::new()?;
            // Resume the run - it will re-attempt failed cases
            rt.block_on(orch.run(Some(resolved_id), limit, false))?;
        }
        Some(Command::Export {
            run_id,
            format,
            output,
            runs_dir,
        }) => {
            let runs_dir = resolve_runs_dir(
                Some(&config),
                runs_dir.map(PathBuf::from),
                Some(&config_path),
            );
            let resolved_id = resolve_run_id(&runs_dir, &run_id)?;
            let store = RunStore::new(runs_dir.join(&resolved_id))?;
            let results = store.load_results()?;
            let agg = aggregate_results(&results);

            let content = match format.as_str() {
                "json" => {
                    let export = serde_json::json!({
                        "run_id": resolved_id,
                        "metrics": agg,
                        "results": results,
                    });
                    serde_json::to_string_pretty(&export)?
                }
                "markdown" | "md" => {
                    let mut md = String::new();
                    md.push_str(&format!("# Run: {resolved_id}\n\n"));
                    md.push_str("## Summary\n\n");
                    md.push_str(&format!("- **Total Cases**: {}\n", agg.total_cases));
                    md.push_str(&format!("- **Accuracy**: {:.1}%\n", agg.accuracy * 100.0));
                    md.push_str(&format!("- **Failed Cases**: {}\n", agg.failed_cases));
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
                    md.push_str(&format!("- **Cost**: ${:.4}\n", agg.total_cost_usd));

                    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
                    if !failed.is_empty() {
                        md.push_str("\n## Failed Cases\n\n");
                        md.push_str("| Dataset | Case ID | Error |\n");
                        md.push_str("|---------|---------|-------|\n");
                        for f in &failed {
                            let err = f
                                .error
                                .as_deref()
                                .unwrap_or("evaluation failed")
                                .replace('|', "\\|");
                            md.push_str(&format!(
                                "| {} | {} | {} |\n",
                                f.dataset_name, f.case_id, err
                            ));
                        }
                    }
                    md
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

/// Resolve a partial run ID to a full run ID by finding matching directories.
fn resolve_run_id(runs_dir: &PathBuf, partial: &str) -> Result<String> {
    // If exact match exists, use it
    if runs_dir.join(partial).exists() {
        return Ok(partial.to_string());
    }

    // Find directories that start with the partial ID
    let entries: Vec<_> = fs::read_dir(runs_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter(|e| e.file_name().to_string_lossy().starts_with(partial))
        .collect();

    match entries.len() {
        0 => Err(anyhow!("no run found matching '{partial}'")),
        1 => Ok(entries[0].file_name().to_string_lossy().to_string()),
        _ => {
            let matches: Vec<_> = entries
                .iter()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            Err(anyhow!(
                "ambiguous run ID '{partial}', matches: {}",
                matches.join(", ")
            ))
        }
    }
}
