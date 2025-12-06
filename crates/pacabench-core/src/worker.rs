//! Worker pool with agent-scoped queues.
//!
//! Each agent has its own async_channel, ensuring cases are handled by the correct runner.
//!
//! Internal types:
//! - [`WorkItem`]: Unit of work dispatched to workers
//! - [`WorkResult`]: Result from processing a work item

use crate::config::{AgentConfig, Config};
use crate::error::{PacabenchError, Result};
use crate::evaluators::{get_evaluator, Evaluator};
use crate::proxy::{ProxyConfig, ProxyServer};
use crate::runner::{CommandRunner, RunnerOutput};
use crate::types::{Case, CaseKey, CaseResult, ErrorType, EvaluationResult, Event, LlmMetrics};
use anyhow::anyhow;
use async_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tracing::{debug, info, warn};

/// A single unit of work: run a case through an agent.
///
/// Work items are created by the case producer and dispatched to workers
/// via agent-scoped queues.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkItem {
    /// ID of the benchmark run this work belongs to.
    pub run_id: String,
    /// Name of the agent to process this case.
    pub agent_name: String,
    /// Name of the dataset containing this case.
    pub dataset_name: String,
    /// Unique ID of the case within the dataset.
    pub case_id: String,
    /// The case data to process.
    pub case: Case,
    /// Current attempt number (1-based, increments on retry).
    pub attempt: u32,
}

impl WorkItem {
    /// Create a retry of this work item with incremented attempt counter.
    pub fn retry(&self) -> Self {
        Self {
            run_id: self.run_id.clone(),
            agent_name: self.agent_name.clone(),
            dataset_name: self.dataset_name.clone(),
            case_id: self.case_id.clone(),
            case: self.case.clone(),
            attempt: self.attempt + 1,
        }
    }

    /// Get the unique key identifying this case.
    pub fn key(&self) -> CaseKey {
        CaseKey::new(&self.agent_name, &self.dataset_name, &self.case_id)
    }
}

/// Result from processing a work item.
///
/// Contains both the runner output and evaluation results, ready to be
/// converted to a [`CaseResult`] for persistence.
#[derive(Clone, Debug)]
pub struct WorkResult {
    /// The original work item that produced this result.
    pub item: WorkItem,
    /// Whether the case passed evaluation.
    pub passed: bool,
    /// Raw output from the runner.
    pub output: Option<String>,
    /// Error message if the runner failed.
    pub error: Option<String>,
    /// Classification of the error.
    pub error_type: ErrorType,
    /// Wall-clock time for the runner, in milliseconds.
    pub duration_ms: f64,
    /// LLM metrics collected by the proxy.
    pub llm_metrics: LlmMetrics,
    /// Evaluation result, if an evaluator was configured.
    pub evaluation: Option<EvaluationResult>,
}

impl WorkResult {
    /// Returns true if this result represents a system or fatal error.
    #[allow(dead_code)]
    pub fn is_error(&self) -> bool {
        self.error_type.is_error()
    }

    /// Convert to a [`CaseResult`] for persistence.
    pub fn to_case_result(&self, timestamp: String) -> CaseResult {
        CaseResult {
            case_id: self.item.case_id.clone(),
            dataset_name: self.item.dataset_name.clone(),
            agent_name: self.item.agent_name.clone(),
            passed: self.passed,
            attempt: self.item.attempt,
            output: self.output.clone(),
            error: self.error.clone(),
            error_type: self.error_type,
            runner_duration_ms: self.duration_ms,
            llm_metrics: self.llm_metrics.clone(),
            timestamp: Some(timestamp),
            f1_score: self.evaluation.as_ref().and_then(|e| e.f1_score),
            f1_passed: self.evaluation.as_ref().map(|e| e.passed),
            judge_passed: self.evaluation.as_ref().map(|e| e.passed),
            judge_reason: self.evaluation.as_ref().and_then(|e| e.reason.clone()),
            judge_metrics: self
                .evaluation
                .as_ref()
                .and_then(|e| e.judge_metrics.clone()),
        }
    }
}

/// Pool of workers that process cases in parallel.
///
/// Each agent has its own queue, ensuring cases are routed to the correct
/// runner. Workers are distributed across agents based on concurrency settings.
pub struct WorkerPool {
    work_txs: HashMap<String, Sender<WorkItem>>,
    result_rx: mpsc::UnboundedReceiver<WorkResult>,
    #[allow(dead_code)]
    event_tx: broadcast::Sender<Event>,
    workers: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    /// Start the worker pool with the given concurrency and configuration.
    ///
    /// Creates agent-scoped queues and spawns workers according to the
    /// concurrency setting (at least one worker per agent).
    pub async fn start(
        concurrency: usize,
        queue_capacity: usize,
        config: &Config,
        event_tx: broadcast::Sender<Event>,
    ) -> Result<Self> {
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        let evaluators: Vec<(String, Arc<dyn Evaluator>)> = config
            .datasets
            .iter()
            .filter_map(|ds| ds.evaluator.as_ref().map(|cfg| (ds, cfg)))
            .map(|(ds, cfg)| {
                get_evaluator(cfg)
                    .map(|e| (ds.name.clone(), e))
                    .map_err(|e| {
                        PacabenchError::evaluation(anyhow!(
                            "building evaluator for dataset {}: {e}",
                            ds.name
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let evaluators = Arc::new(evaluators);

        let mut work_txs = HashMap::new();
        let mut work_rxs = HashMap::new();
        for agent in &config.agents {
            let (tx, rx) = async_channel::bounded(queue_capacity.max(1));
            work_txs.insert(agent.name.clone(), tx);
            work_rxs.insert(agent.name.clone(), rx);
        }

        let agent_count = config.agents.len().max(1);
        let total_workers = concurrency.max(agent_count);
        // Ensure at least one worker per agent, then fan out remaining capacity round-robin.
        let mut distribution = vec![1usize; agent_count];
        for i in 0..(total_workers - agent_count) {
            distribution[i % agent_count] += 1;
        }

        let mut workers = Vec::with_capacity(total_workers);
        let mut worker_id = 0u64;

        for (idx, agent) in config.agents.iter().enumerate() {
            let Some(rx) = work_rxs.get(&agent.name) else {
                continue;
            };

            for _ in 0..distribution[idx] {
                let worker_cfg = WorkerConfig {
                    id: worker_id,
                    agent: agent.clone(),
                    proxy_enabled: config.global.proxy.enabled,
                    proxy_base_url: config.global.proxy.base_url.clone(),
                    timeout_seconds: config.global.timeout_seconds,
                    root_dir: config.root_dir.clone(),
                };

                let handle = spawn_worker(
                    worker_cfg,
                    rx.clone(),
                    result_tx.clone(),
                    event_tx.clone(),
                    evaluators.clone(),
                )
                .await?;

                workers.push(handle);
                worker_id += 1;
            }
        }

        Ok(Self {
            work_txs,
            result_rx,
            event_tx,
            workers,
        })
    }

    /// Push a work item to the queue.
    pub async fn push(&self, item: WorkItem) {
        if let Some(tx) = self.work_txs.get(&item.agent_name) {
            let _ = tx.send(item).await;
        } else {
            warn!(
                agent = %item.agent_name,
                "no worker queue found for agent"
            );
        }
    }

    /// Receive the next completed work result.
    ///
    /// Returns `None` when all workers have stopped and the result channel is empty.
    pub async fn recv(&mut self) -> Option<WorkResult> {
        self.result_rx.recv().await
    }

    /// Subscribe to events.
    #[allow(dead_code)]
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.event_tx.subscribe()
    }

    /// Check if work queue is empty.
    #[allow(dead_code)]
    pub fn is_work_queue_empty(&self) -> bool {
        self.work_txs.values().all(Sender::is_empty)
    }

    /// Close the work queue and wait for workers to finish.
    pub async fn shutdown(self) {
        for tx in self.work_txs.values() {
            tx.close();
        }
        for handle in self.workers {
            let _ = handle.await;
        }
    }
}

#[derive(Clone)]
struct WorkerConfig {
    id: u64,
    agent: AgentConfig,
    proxy_enabled: bool,
    proxy_base_url: Option<String>,
    timeout_seconds: f64,
    root_dir: PathBuf,
}

async fn spawn_worker(
    config: WorkerConfig,
    work_rx: Receiver<WorkItem>,
    result_tx: mpsc::UnboundedSender<WorkResult>,
    event_tx: broadcast::Sender<Event>,
    evaluators: Arc<Vec<(String, Arc<dyn Evaluator>)>>,
) -> Result<JoinHandle<()>> {
    // Create proxy
    let proxy = if config.proxy_enabled {
        let upstream = config
            .proxy_base_url
            .clone()
            .unwrap_or_else(|| crate::utils::DEFAULT_OPENAI_BASE_URL.to_string());
        Some(
            ProxyServer::start(ProxyConfig {
                port: 0,
                upstream_base_url: upstream,
                api_key: std::env::var("OPENAI_API_KEY").ok(),
            })
            .await?,
        )
    } else {
        None
    };

    let proxy_url = proxy.as_ref().map(|p| format!("http://{}/v1", p.addr));

    // Create runner
    let mut runner = CommandRunner::new(
        config.agent.clone(),
        proxy_url,
        None,
        Some(config.root_dir.clone()),
    );
    runner.start().await?;

    let handle = tokio::spawn(async move {
        worker_loop(
            config, work_rx, result_tx, event_tx, runner, proxy, evaluators,
        )
        .await;
    });

    Ok(handle)
}

async fn worker_loop(
    config: WorkerConfig,
    work_rx: Receiver<WorkItem>,
    result_tx: mpsc::UnboundedSender<WorkResult>,
    event_tx: broadcast::Sender<Event>,
    mut runner: CommandRunner,
    proxy: Option<ProxyServer>,
    evaluators: Arc<Vec<(String, Arc<dyn Evaluator>)>>,
) {
    info!(worker_id = config.id, "Worker started");

    while let Ok(item) = work_rx.recv().await {
        debug!(worker_id = config.id, case_id = %item.case_id, "Processing case");

        // Emit start event
        let _ = event_tx.send(Event::CaseStarted {
            run_id: item.run_id.clone(),
            case_id: item.case_id.clone(),
            agent: item.agent_name.clone(),
            dataset: item.dataset_name.clone(),
            attempt: item.attempt,
        });

        // Clear proxy metrics before running
        if let Some(ref p) = proxy {
            let _ = p.metrics.snapshot_and_clear();
        }

        // Run the case
        let start = Instant::now();
        let result = timeout(
            Duration::from_secs_f64(config.timeout_seconds),
            runner.run_case(&item.case),
        )
        .await;

        let (output, error, error_type, duration_ms) = match result {
            Ok(Ok(runner_output)) => {
                let duration = if runner_output.duration_ms > 0.0 {
                    runner_output.duration_ms
                } else {
                    start.elapsed().as_millis() as f64
                };
                (
                    runner_output.output,
                    runner_output.error,
                    runner_output.error_type,
                    duration,
                )
            }
            Ok(Err(e)) => (
                None,
                Some(format!("Runner error: {}", e)),
                ErrorType::SystemFailure,
                start.elapsed().as_millis() as f64,
            ),
            Err(_) => (
                None,
                Some("Timeout".to_string()),
                ErrorType::SystemFailure,
                start.elapsed().as_millis() as f64,
            ),
        };

        // Collect proxy metrics
        let llm_metrics = proxy
            .as_ref()
            .map(|p| p.metrics.snapshot_and_clear())
            .unwrap_or_default();

        // Run evaluation if we have output
        let evaluation = if output.is_some() && error.is_none() {
            run_evaluation(&item, &output, &evaluators).await
        } else {
            None
        };

        let passed = evaluation
            .as_ref()
            .map(|e| e.passed)
            .unwrap_or(error.is_none());

        // Send result back to benchmark
        let work_result = WorkResult {
            item: item.clone(),
            passed,
            output,
            error,
            error_type,
            duration_ms,
            llm_metrics: llm_metrics.clone(),
            evaluation,
        };

        if result_tx.send(work_result).is_err() {
            warn!(worker_id = config.id, "Result channel closed");
            break;
        }
    }

    let _ = runner.stop().await;
    info!(worker_id = config.id, "Worker stopped");
}

async fn run_evaluation(
    item: &WorkItem,
    output: &Option<String>,
    evaluators: &[(String, Arc<dyn Evaluator>)],
) -> Option<EvaluationResult> {
    let evaluator = evaluators
        .iter()
        .find(|(ds, _)| ds == &item.dataset_name)
        .map(|(_, e)| e.clone())?;

    let runner_output = RunnerOutput {
        output: output.clone(),
        error: None,
        error_type: ErrorType::None,
        duration_ms: 0.0,
        error_traceback: None,
    };

    Some(evaluator.evaluate(&item.case, &runner_output).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_work_item_retry() {
        let case = Case {
            case_id: "test-1".into(),
            dataset_name: "test-ds".into(),
            input: "question".into(),
            expected: Some("answer".into()),
            history: vec![],
            metadata: HashMap::new(),
        };

        let item = WorkItem {
            run_id: "run-123".into(),
            agent_name: "agent-1".into(),
            dataset_name: case.dataset_name.clone(),
            case_id: case.case_id.clone(),
            case,
            attempt: 1,
        };
        assert_eq!(item.attempt, 1);

        let retry = item.retry();
        assert_eq!(retry.attempt, 2);
        assert_eq!(retry.case_id, "test-1");
    }
}
