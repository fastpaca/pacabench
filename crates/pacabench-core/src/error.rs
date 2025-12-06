//! Unified error types for PacaBench.
//!
//! This module provides a comprehensive error hierarchy using `thiserror`
//! for all operations in the benchmark pipeline.

use thiserror::Error;

/// The main error type for PacaBench operations.
#[derive(Debug, Error)]
pub enum PacabenchError {
    /// Configuration loading or validation failed.
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),

    /// Dataset loading failed.
    #[error("dataset error in '{dataset}': {source}")]
    Dataset {
        dataset: String,
        #[source]
        source: anyhow::Error,
    },

    /// Runner (agent) execution failed.
    #[error("runner error for agent '{agent}': {source}")]
    Runner {
        agent: String,
        #[source]
        source: anyhow::Error,
    },

    /// Proxy server error.
    #[error("proxy error: {0}")]
    Proxy(#[source] std::io::Error),

    /// Evaluation failed.
    #[error("evaluation error: {0}")]
    Evaluation(#[source] anyhow::Error),

    /// Persistence (file I/O) error.
    #[error("persistence error: {0}")]
    Persistence(#[source] std::io::Error),

    /// Circuit breaker tripped due to high error rate.
    #[error("circuit breaker tripped at {error_ratio:.1}% error rate")]
    CircuitTripped { error_ratio: f64 },

    /// Process spawning or management failed.
    #[error("process error: {0}")]
    Process(#[source] std::io::Error),

    /// Channel communication failed.
    #[error("channel closed unexpectedly")]
    ChannelClosed,

    /// Timeout exceeded.
    #[error("operation timed out after {seconds:.1}s")]
    Timeout { seconds: f64 },

    /// Generic internal error.
    #[error("{0}")]
    Internal(#[from] anyhow::Error),
}

/// Result type alias using `PacabenchError`.
pub type Result<T> = std::result::Result<T, PacabenchError>;

impl From<serde_json::Error> for PacabenchError {
    fn from(err: serde_json::Error) -> Self {
        PacabenchError::Internal(err.into())
    }
}

impl PacabenchError {
    /// Create a dataset error with context.
    pub fn dataset(name: impl Into<String>, source: impl Into<anyhow::Error>) -> Self {
        Self::Dataset {
            dataset: name.into(),
            source: source.into(),
        }
    }

    /// Create a runner error with context.
    pub fn runner(agent: impl Into<String>, source: impl Into<anyhow::Error>) -> Self {
        Self::Runner {
            agent: agent.into(),
            source: source.into(),
        }
    }

    /// Create an evaluation error.
    pub fn evaluation(source: impl Into<anyhow::Error>) -> Self {
        Self::Evaluation(source.into())
    }
}
