//! Event bus for observability and checkpointing.
//!
//! Provides a publish-subscribe mechanism for benchmark events.
//! Events flow from producers (orchestrator, proxy, runners) to
//! subscribers (progress reporters, metrics aggregators, loggers).

use crate::state::RunStateKind;
use crate::types::{AggregatedMetrics, CaseResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Events emitted during benchmark execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PacabenchEvent {
    /// Run has started.
    RunStarted {
        run_id: String,
        total_cases: u64,
        resuming: bool,
    },

    /// Run state has changed.
    StateTransition {
        from: RunStateKind,
        to: RunStateKind,
    },

    /// A case is about to be executed.
    CaseStarted {
        case_id: String,
        agent_name: String,
        dataset_name: String,
    },

    /// A case has completed (pass or fail).
    CaseCompleted {
        case_id: String,
        agent_name: String,
        dataset_name: String,
        passed: bool,
        is_system_error: bool,
        duration_ms: f64,
        cost_usd: f64,
    },

    /// An LLM request was made through the proxy.
    LlmRequest {
        latency_ms: f64,
        model: Option<String>,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        cost_usd: f64,
    },

    /// Circuit breaker has tripped.
    CircuitTripped { error_ratio: f64 },

    /// Run has completed.
    RunCompleted {
        run_id: String,
        total_cases: u64,
        passed_cases: u64,
        failed_cases: u64,
        total_cost_usd: f64,
        circuit_tripped: bool,
    },

    /// Aggregated metrics are available.
    MetricsReady { metrics: AggregatedMetrics },

    /// A system error occurred.
    SystemError {
        agent_name: Option<String>,
        case_id: Option<String>,
        error: String,
    },
}

/// Handle for sending events to the bus.
#[derive(Clone, Debug)]
pub struct EventSender {
    senders: Arc<Vec<mpsc::UnboundedSender<PacabenchEvent>>>,
}

impl EventSender {
    /// Emit an event to all subscribers.
    pub fn emit(&self, event: PacabenchEvent) {
        for sender in self.senders.iter() {
            // Ignore send errors - subscriber may have dropped
            let _ = sender.send(event.clone());
        }
    }

    /// Convenience method to emit a case result as an event.
    pub fn emit_case_completed(&self, result: &CaseResult, cost_usd: f64) {
        self.emit(PacabenchEvent::CaseCompleted {
            case_id: result.case_id.clone(),
            agent_name: result.agent_name.clone(),
            dataset_name: result.dataset_name.clone(),
            passed: result.passed,
            is_system_error: matches!(
                result.error_type,
                crate::types::ErrorType::SystemFailure | crate::types::ErrorType::FatalError
            ),
            duration_ms: result.runner_duration_ms,
            cost_usd,
        });
    }
}

/// Event bus for broadcasting events to multiple subscribers.
pub struct EventBus {
    senders: Vec<mpsc::UnboundedSender<PacabenchEvent>>,
}

impl EventBus {
    /// Create a new empty event bus.
    pub fn new() -> Self {
        Self {
            senders: Vec::new(),
        }
    }

    /// Subscribe to events, returning a receiver.
    pub fn subscribe(&mut self) -> mpsc::UnboundedReceiver<PacabenchEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.senders.push(tx);
        rx
    }

    /// Get an event sender handle that can be cloned and shared.
    pub fn sender(&self) -> EventSender {
        EventSender {
            senders: Arc::new(self.senders.clone()),
        }
    }

    /// Emit an event to all subscribers (convenience method).
    pub fn emit(&self, event: PacabenchEvent) {
        for sender in &self.senders {
            let _ = sender.send(event.clone());
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Token usage information for LLM requests.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}

