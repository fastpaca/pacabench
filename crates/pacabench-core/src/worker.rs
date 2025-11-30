//! Worker pool with work-stealing MPMC queue.
//!
//! Workers pop from a shared async_channel, process cases, and send results back.

use crate::config::{AgentConfig, Config};
use crate::evaluators::{get_evaluator, Evaluator};
use crate::protocol::{Event, WorkItem, WorkResult};
use crate::proxy::{ProxyConfig, ProxyServer};
use crate::runner::CommandRunner;
use crate::types::{ErrorType, EvaluationResult, RunnerOutput};
use anyhow::Result;
use async_channel::{Receiver, Sender};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tracing::{debug, info, warn};

// ============================================================================
// WORKER POOL
// ============================================================================

pub struct WorkerPool {
    work_tx: Sender<WorkItem>,
    result_rx: mpsc::UnboundedReceiver<WorkResult>,
    #[allow(dead_code)]
    event_tx: broadcast::Sender<Event>,
    workers: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    pub async fn start(
        concurrency: usize,
        config: &Config,
        event_tx: broadcast::Sender<Event>,
    ) -> Result<Self> {
        let (work_tx, work_rx) = async_channel::unbounded();
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        let evaluators: Vec<(String, Arc<dyn Evaluator>)> = config
            .datasets
            .iter()
            .filter_map(|ds| {
                ds.evaluator
                    .as_ref()
                    .and_then(|cfg| get_evaluator(cfg).ok())
                    .map(|e| (ds.name.clone(), e))
            })
            .collect();
        let evaluators = Arc::new(evaluators);

        let mut workers = Vec::with_capacity(concurrency);

        for i in 0..concurrency {
            let agent_idx = i % config.agents.len();
            let agent = config.agents[agent_idx].clone();

            let worker_cfg = WorkerConfig {
                id: i as u64,
                agent,
                proxy_enabled: config.global.proxy.enabled,
                proxy_base_url: config.global.proxy.base_url.clone(),
                timeout_seconds: config.global.timeout_seconds,
                root_dir: config.root_dir.clone(),
            };

            let handle = spawn_worker(
                worker_cfg,
                work_rx.clone(),
                result_tx.clone(),
                event_tx.clone(),
                evaluators.clone(),
            )
            .await?;

            workers.push(handle);
        }

        Ok(Self {
            work_tx,
            result_rx,
            event_tx,
            workers,
        })
    }

    /// Push a work item to the queue.
    pub async fn push(&self, item: WorkItem) {
        // Unbounded channel - send never fails unless closed
        let _ = self.work_tx.send(item).await;
    }

    /// Push all work items to the queue.
    pub async fn push_batch(&self, items: Vec<WorkItem>) {
        for item in items {
            let _ = self.work_tx.send(item).await;
        }
    }

    /// Receive the next result (or None if channel closed).
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
        self.work_tx.is_empty()
    }

    /// Close the work queue and wait for workers to finish.
    pub async fn shutdown(self) {
        self.work_tx.close();
        for handle in self.workers {
            let _ = handle.await;
        }
    }
}

// ============================================================================
// WORKER
// ============================================================================

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
            .unwrap_or_else(|| "https://api.openai.com".to_string());
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
        worker_loop(config, work_rx, result_tx, event_tx, runner, proxy, evaluators).await;
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

        let passed = evaluation.as_ref().map(|e| e.passed).unwrap_or(error.is_none());

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
