//! Concurrent orchestration: load datasets, run agents in parallel, and persist results.

use crate::config::BenchmarkConfig;
use crate::datasets::{get_dataset_loader, DatasetContext, DatasetLoader};
use crate::evaluators::get_evaluator;
use crate::metrics::aggregate_results;
use crate::persistence::{iso_timestamp_now, ErrorEntry};
use crate::proxy::{MetricEntry, ProxyConfig, ProxyServer};
use crate::reporter::{AgentProgress, NullReporter, ProgressEvent, ProgressReporter};
use crate::run_manager::RunManager;
use crate::runner::CommandRunner;
use crate::types::{Case, CaseResult, ErrorType, RunnerOutput};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::{timeout, Duration};

pub struct Orchestrator {
    config: BenchmarkConfig,
    root_dir: PathBuf,
    cache_dir: PathBuf,
    runs_dir: PathBuf,
    reporter: Arc<dyn ProgressReporter>,
    config_path: Option<PathBuf>,
}

struct WorkItem {
    agent_idx: usize,
    dataset_idx: usize,
    case: Case,
}

struct CircuitBreaker {
    consecutive_failures: AtomicU64,
    total_cases: AtomicU64,
    total_failures: AtomicU64,
    tripped: AtomicBool,
    error_ratio_threshold: Option<f64>,
    min_cases: u64,
}

impl CircuitBreaker {
    fn new(error_ratio: Option<f64>, min_cases: usize) -> Self {
        Self {
            consecutive_failures: AtomicU64::new(0),
            total_cases: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            tripped: AtomicBool::new(false),
            error_ratio_threshold: error_ratio,
            min_cases: min_cases as u64,
        }
    }

    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
        self.total_cases.fetch_add(1, Ordering::SeqCst);
    }

    fn record_failure(&self) -> bool {
        self.consecutive_failures.fetch_add(1, Ordering::SeqCst);
        self.total_cases.fetch_add(1, Ordering::SeqCst);
        self.total_failures.fetch_add(1, Ordering::SeqCst);

        if let Some(threshold) = self.error_ratio_threshold {
            let total = self.total_cases.load(Ordering::SeqCst);
            let failures = self.total_failures.load(Ordering::SeqCst);
            if total >= self.min_cases && (failures as f64 / total as f64) > threshold {
                let was_tripped = self.tripped.swap(true, Ordering::SeqCst);
                return !was_tripped; // Return true if we just tripped it
            }
        }
        false
    }

    fn error_ratio(&self) -> f64 {
        let total = self.total_cases.load(Ordering::SeqCst);
        let failures = self.total_failures.load(Ordering::SeqCst);
        if total == 0 {
            0.0
        } else {
            failures as f64 / total as f64
        }
    }

    fn is_tripped(&self) -> bool {
        self.tripped.load(Ordering::SeqCst)
    }
}

impl Orchestrator {
    pub fn new(
        config: BenchmarkConfig,
        root_dir: PathBuf,
        cache_dir: PathBuf,
        runs_dir: PathBuf,
    ) -> Self {
        Self {
            config,
            root_dir,
            cache_dir,
            runs_dir,
            reporter: Arc::new(NullReporter),
            config_path: None,
        }
    }

    /// Set a custom progress reporter.
    pub fn with_reporter(mut self, reporter: Arc<dyn ProgressReporter>) -> Self {
        self.reporter = reporter;
        self
    }

    /// Set the config path for copying to run directory.
    pub fn with_config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    pub async fn run(
        &self,
        run_id: Option<String>,
        limit: Option<usize>,
        force_new: bool,
    ) -> Result<()> {
        let rm = Arc::new(Mutex::new(RunManager::new(
            &self.config,
            self.runs_dir.clone(),
            run_id,
            force_new,
            self.config_path.clone(),
        )?));

        let datasets = self.load_all_datasets(limit)?;

        // Build work items, skipping already-completed cases
        let mut work_items = Vec::new();
        {
            let rm_guard = rm.lock().await;
            for (agent_idx, agent) in self.config.agents.iter().enumerate() {
                for (dataset_idx, (ds, cases)) in datasets.iter().enumerate() {
                    for case in cases {
                        // Skip only if case already passed (for resume/retry)
                        if rm_guard.resuming
                            && rm_guard.is_passed(&agent.name, &ds.name, &case.case_id)
                        {
                            continue;
                        }
                        work_items.push(WorkItem {
                            agent_idx,
                            dataset_idx,
                            case: case.clone(),
                        });
                    }
                }
            }
        }

        let total_cases = work_items.len() as u64 + {
            let g = rm.lock().await;
            g.completed_count() as u64
        };

        let resuming = {
            let rm_guard = rm.lock().await;
            rm_guard.resuming
        };
        let completed_before = {
            let rm_guard = rm.lock().await;
            rm_guard.completed_count() as u64
        };

        {
            let mut rm_guard = rm.lock().await;
            rm_guard.set_total_cases(total_cases)?;
            rm_guard.initialize_metadata()?;
        }

        let run_id = {
            let rm_guard = rm.lock().await;
            rm_guard.run_id.clone()
        };

        // Build per-agent progress information
        let agents = {
            let mut agent_progress: HashMap<String, AgentProgress> = HashMap::new();

            // Initialize with all agents
            for agent in &self.config.agents {
                agent_progress.insert(agent.name.clone(), AgentProgress::default());
            }

            // Count pending cases per agent
            for item in &work_items {
                let agent_name = &self.config.agents[item.agent_idx].name;
                if let Some(ap) = agent_progress.get_mut(agent_name) {
                    ap.total_cases += 1;
                }
            }

            // Get completed cases per agent from RunManager
            let rm_guard = rm.lock().await;
            for (agent_name, ap) in agent_progress.iter_mut() {
                let completed = rm_guard.completed_count_for_agent(agent_name) as u64;
                ap.completed_cases = completed;
                ap.total_cases += completed;
            }

            agent_progress
        };

        self.reporter.report(ProgressEvent::RunStarted {
            run_id: run_id.clone(),
            total_cases,
            resuming,
            completed_cases: completed_before,
            agents,
        });

        if work_items.is_empty() {
            let rm_guard = rm.lock().await;
            rm_guard.mark_completed(false)?;
            self.reporter.report(ProgressEvent::RunCompleted {
                run_id,
                total_cases,
                passed_cases: rm_guard.passed_count() as u64,
                failed_cases: total_cases.saturating_sub(rm_guard.passed_count() as u64),
                total_cost_usd: 0.0,
                circuit_tripped: false,
            });
            return Ok(());
        }

        // Start proxy if enabled
        let proxy = if self.config.config.proxy.enabled {
            // Resolve upstream URL from config or provider
            let upstream_base_url =
                self.config.config.proxy.base_url.clone().or_else(|| {
                    match self.config.config.proxy.provider.as_str() {
                        "openai" => Some("https://api.openai.com".to_string()),
                        "anthropic" => Some("https://api.anthropic.com".to_string()),
                        "azure" => std::env::var("AZURE_OPENAI_ENDPOINT").ok(),
                        _ => None,
                    }
                });
            Some(
                ProxyServer::start(ProxyConfig {
                    port: 0,
                    upstream_base_url,
                    api_key: std::env::var("OPENAI_API_KEY").ok(),
                })
                .await?,
            )
        } else {
            None
        };
        let proxy_url = proxy.as_ref().map(|p| format!("http://{}/v1", p.addr));
        let metrics_handle = proxy.as_ref().map(|p| p.metrics.clone());

        // Create runner pool per agent
        let mut runners: Vec<Arc<Mutex<CommandRunner>>> = Vec::new();
        for agent in &self.config.agents {
            let runner = CommandRunner::new(
                agent.clone(),
                proxy_url.clone(),
                None,
                Some(self.root_dir.clone()),
            );
            runners.push(Arc::new(Mutex::new(runner)));
        }

        // Start all runners
        for runner in &runners {
            let mut r = runner.lock().await;
            r.start().await?;
        }

        // Build evaluators per dataset
        let evaluators: Vec<_> = datasets
            .iter()
            .map(|(ds, _)| {
                ds.evaluator
                    .as_ref()
                    .and_then(|cfg| get_evaluator(cfg).ok())
            })
            .collect();

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            self.config.config.circuit_breaker_error_ratio,
            self.config.config.circuit_breaker_min_cases,
        ));

        let semaphore = Arc::new(Semaphore::new(self.config.config.concurrency));
        let timeout_secs = self.config.config.timeout_seconds;
        let max_retries = self.config.config.max_retries;
        let recycle_interval = self.config.config.worker_recycle_interval;
        let recycle_counts = Arc::new(
            (0..self.config.agents.len())
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        );

        let mut handles = Vec::new();

        for item in work_items {
            let permit = semaphore.clone().acquire_owned().await?;
            let runner = runners[item.agent_idx].clone();
            let rm = rm.clone();
            let metrics_handle = metrics_handle.clone();
            let circuit_breaker = circuit_breaker.clone();
            let agent_name = self.config.agents[item.agent_idx].name.clone();
            let ds_name = datasets[item.dataset_idx].0.name.clone();
            let evaluator = evaluators[item.dataset_idx].clone();
            let eval_kind = datasets[item.dataset_idx]
                .0
                .evaluator
                .as_ref()
                .map(|e| e.r#type.clone());
            let recycle_counts = recycle_counts.clone();
            let reporter = self.reporter.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;

                if circuit_breaker.is_tripped() {
                    return;
                }

                let case = item.case;

                for retry in 0..=max_retries {
                    if circuit_breaker.is_tripped() {
                        break;
                    }

                    // Clear metrics before each attempt
                    if let Some(m) = &metrics_handle {
                        let _ = m.snapshot_and_clear().await;
                    }

                    let attempt = {
                        let rm_guard = rm.lock().await;
                        rm_guard.get_next_attempt(&agent_name, &ds_name, &case.case_id)
                    };

                    let start = Instant::now();
                    let out: Result<RunnerOutput> = {
                        let mut runner_guard = runner.lock().await;
                        timeout(
                            Duration::from_secs_f64(timeout_secs),
                            runner_guard.run_case(&case),
                        )
                        .await
                        .map_err(|_| anyhow!("timeout"))
                        .and_then(|res| res)
                    };

                    let mut completed_attempt = false;
                    match out {
                        Ok(mut ro) => {
                            if ro.duration_ms == 0.0 {
                                ro.duration_ms = start.elapsed().as_millis() as f64;
                            }

                            let (eval, kind) = if let Some(ev) = evaluator.as_ref() {
                                let res = ev.evaluate(&case, &ro);
                                (Some(res), eval_kind.clone())
                            } else {
                                (None, None)
                            };

                            let merged_metrics = if let Some(mh) = &metrics_handle {
                                merge_proxy_entries(&mh.snapshot_and_clear().await)
                            } else {
                                HashMap::new()
                            };

                            let result = case_result_from_output(
                                &case,
                                &agent_name,
                                &ds_name,
                                attempt,
                                ro.clone(),
                                eval.as_ref(),
                                kind.as_deref(),
                                Some(merged_metrics),
                            );

                            let passed = result.passed;
                            let is_system_error = matches!(
                                result.error_type,
                                ErrorType::SystemFailure | ErrorType::FatalError
                            );
                            let case_cost = result
                                .llm_metrics
                                .get("llm_total_cost_usd")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0)
                                + result.judge_cost_usd.unwrap_or(0.0);

                            {
                                let mut rm_guard = rm.lock().await;
                                let _ = rm_guard.append_result(&result);
                            }

                            if is_system_error {
                                if circuit_breaker.record_failure() {
                                    reporter.report(ProgressEvent::CircuitTripped {
                                        error_ratio: circuit_breaker.error_ratio(),
                                    });
                                }
                            } else {
                                circuit_breaker.record_success();
                            }

                            // Determine if this is the final attempt for this case
                            let is_final_attempt =
                                passed || is_system_error || retry >= max_retries;

                            // Only report CaseCompleted when no more retries will occur
                            if is_final_attempt {
                                reporter.report(ProgressEvent::CaseCompleted {
                                    case_id: case.case_id.clone(),
                                    agent_name: agent_name.clone(),
                                    dataset_name: ds_name.clone(),
                                    passed,
                                    is_system_error,
                                    duration_ms: result.runner_duration_ms,
                                    cost_usd: case_cost,
                                });
                                completed_attempt = true;
                            } else {
                                // Retry after backoff
                                tokio::time::sleep(Duration::from_millis(100 * (retry as u64 + 1)))
                                    .await;
                            }
                        }
                        Err(err) => {
                            if circuit_breaker.record_failure() {
                                reporter.report(ProgressEvent::CircuitTripped {
                                    error_ratio: circuit_breaker.error_ratio(),
                                });
                            }

                            if retry == max_retries {
                                let mut rm_guard = rm.lock().await;
                                let _ = rm_guard.append_error(&ErrorEntry {
                                    timestamp: iso_timestamp_now(),
                                    error_type: ErrorType::SystemFailure,
                                    agent_name: Some(agent_name.clone()),
                                    dataset_name: Some(ds_name.clone()),
                                    case_id: Some(case.case_id.clone()),
                                    error: Some(err.to_string()),
                                });

                                reporter.report(ProgressEvent::CaseCompleted {
                                    case_id: case.case_id.clone(),
                                    agent_name: agent_name.clone(),
                                    dataset_name: ds_name.clone(),
                                    passed: false,
                                    is_system_error: true,
                                    duration_ms: 0.0,
                                    cost_usd: 0.0,
                                });
                                completed_attempt = true;
                            } else {
                                tokio::time::sleep(Duration::from_millis(100 * (retry as u64 + 1)))
                                    .await;
                            }
                        }
                    }

                    if recycle_interval > 0 {
                        let count =
                            recycle_counts[item.agent_idx].fetch_add(1, Ordering::SeqCst) + 1;
                        if count >= recycle_interval {
                            recycle_counts[item.agent_idx].store(0, Ordering::SeqCst);
                            let mut runner_guard = runner.lock().await;
                            let _ = runner_guard.stop().await;
                            let _ = runner_guard.start().await;
                        }
                    }

                    if completed_attempt {
                        return;
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let _ = handle.await;
        }

        // Stop runners
        for runner in &runners {
            let mut r = runner.lock().await;
            let _ = r.stop().await;
        }

        // Stop proxy
        if let Some(p) = proxy {
            p.stop().await;
        }

        let failed = circuit_breaker.is_tripped();
        let (passed_count, total_cost) = {
            let rm_guard = rm.lock().await;
            rm_guard.mark_completed(failed)?;
            let results = self.runs_dir.join(&run_id);
            let store = crate::persistence::RunStore::new(&results).ok();
            let results = store
                .and_then(|s| s.load_results().ok())
                .unwrap_or_default();
            let agg = aggregate_results(&results);
            (
                agg.total_cases - agg.failed_cases,
                agg.total_cost_usd + agg.total_judge_cost_usd,
            )
        };

        self.reporter.report(ProgressEvent::RunCompleted {
            run_id,
            total_cases,
            passed_cases: passed_count,
            failed_cases: total_cases.saturating_sub(passed_count),
            total_cost_usd: total_cost,
            circuit_tripped: failed,
        });

        Ok(())
    }

    fn dataset_loader(&self, ds: &crate::config::DatasetConfig) -> Result<Box<dyn DatasetLoader>> {
        let ctx = DatasetContext {
            root_dir: self.root_dir.clone(),
            cache_dir: self.cache_dir.clone(),
        };
        get_dataset_loader(ds.clone(), ctx).map_err(|e| anyhow!(e))
    }

    pub fn aggregate_latest(&self, run_id: &str) -> Result<crate::types::AggregatedMetrics> {
        let run_dir = self.runs_dir.join(run_id);
        let store = crate::persistence::RunStore::new(&run_dir)?;
        let results = store.load_results()?;
        Ok(aggregate_results(&results))
    }

    fn load_all_datasets(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<(crate::config::DatasetConfig, Vec<Case>)>> {
        let mut out = Vec::new();
        for ds in &self.config.datasets {
            let loader = self.dataset_loader(ds)?;
            let cases = loader.load(limit)?;
            out.push((ds.clone(), cases));
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn case_result_from_output(
    case: &Case,
    agent_name: &str,
    dataset_name: &str,
    attempt: u32,
    output: RunnerOutput,
    eval: Option<&crate::types::EvaluationResult>,
    eval_kind: Option<&str>,
    proxy_metrics: Option<HashMap<String, serde_json::Value>>,
) -> CaseResult {
    let mut llm_metrics = output.metrics.unwrap_or_default();
    if let Some(pm) = proxy_metrics {
        for (k, v) in pm {
            llm_metrics.entry(k).or_insert(v);
        }
    }
    CaseResult {
        case_id: case.case_id.clone(),
        dataset_name: dataset_name.to_string(),
        agent_name: agent_name.to_string(),
        passed: eval.map(|e| e.passed).unwrap_or(output.error.is_none()),
        output: output.output.clone(),
        error: output.error.clone(),
        error_type: output.error_type.clone(),
        runner_duration_ms: output.duration_ms,
        llm_metrics,
        attempt,
        timestamp: Some(iso_timestamp_now()),
        f1_score: if matches!(eval_kind, Some("f1")) {
            eval.map(|e| e.score)
        } else {
            None
        },
        f1_passed: if matches!(eval_kind, Some("f1")) {
            eval.map(|e| e.passed)
        } else {
            None
        },
        judge_passed: if matches!(eval_kind, Some("llm_judge")) {
            eval.map(|e| e.passed)
        } else {
            None
        },
        judge_reason: eval.and_then(|e| e.reason.clone()),
        judge_metrics: eval.map(|e| e.metrics.clone()).unwrap_or_default(),
        judge_cost_usd: eval.and_then(|e| e.cost_usd),
        extra: HashMap::new(),
    }
}

fn merge_proxy_entries(entries: &[MetricEntry]) -> HashMap<String, serde_json::Value> {
    let mut latencies = Vec::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    let mut cached_tokens: u64 = 0;
    let mut total_cost: f64 = 0.0;
    let call_count = entries.len() as u64;

    for e in entries {
        latencies.push(e.latency_ms);

        let prompt = e
            .usage
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let completion = e
            .usage
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached = e
            .usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        input_tokens += prompt;
        output_tokens += completion;
        cached_tokens += cached;

        // Calculate cost if model is known
        if let Some(model) = &e.model {
            total_cost += crate::pricing::calculate_cost(model, prompt, completion, cached);
        }
    }

    let mut out = HashMap::new();
    out.insert("llm_call_count".into(), serde_json::Value::from(call_count));
    out.insert(
        "llm_latency_ms".into(),
        serde_json::Value::Array(latencies.into_iter().map(serde_json::Value::from).collect()),
    );
    out.insert(
        "llm_input_tokens".into(),
        serde_json::Value::from(input_tokens),
    );
    out.insert(
        "llm_output_tokens".into(),
        serde_json::Value::from(output_tokens),
    );
    out.insert(
        "llm_cached_tokens".into(),
        serde_json::Value::from(cached_tokens),
    );
    out.insert(
        "llm_total_cost_usd".into(),
        serde_json::Value::from(total_cost),
    );
    out
}
