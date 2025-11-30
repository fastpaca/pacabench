//! PacaBench Core - LLM agent benchmarking library.
//!
//! # Quick Start
//!
//! ```ignore
//! use pacabench_core::{Benchmark, Event, Command};
//!
//! let bench = Benchmark::from_config_file("pacabench.yaml")?;
//!
//! // Subscribe to events
//! let events = bench.subscribe();
//! tokio::spawn(async move {
//!     while let Some(event) = events.recv().await {
//!         match event {
//!             Event::CaseCompleted { passed, .. } => println!("Case: {}", if passed { "✓" } else { "✗" }),
//!             Event::RunCompleted { metrics, .. } => println!("Done: {:.1}% accuracy", metrics.accuracy * 100.0),
//!             _ => {}
//!         }
//!     }
//! });
//!
//! // Run or control
//! let result = bench.run(None, None).await?;
//! // bench.send(Command::Stop { reason: "user cancelled".into() });
//! ```
//!
//! # Public API
//!
//! - [`Benchmark`] - Main entry point for running benchmarks
//! - [`Event`] - Events emitted during execution (subscribe to observe)
//! - [`Command`] - Commands to control execution (stop, abort)
//! - [`Config`](config::BenchmarkConfig) - Configuration
//! - [`RunResult`](benchmark::RunResult) - Result of a benchmark run

// Public API - the primary interface
pub mod benchmark;
pub use benchmark::{from_config_file, Benchmark, RunResult};

// Protocol - events and commands (public)
pub mod protocol;
pub use protocol::{Command, Event};

// Configuration (public)
pub mod config;

// Types (public)
pub mod types;
pub use types::{
    AggregatedMetrics, Case, CaseKey, CaseResult, ErrorType, EvaluationResult, JudgeMetrics,
    LlmMetrics, RunStatus, RunnerOutput,
};

// Error types (public)
pub mod error;

// Persistence (public for CLI)
pub mod persistence;

// Metrics aggregation (public for CLI)
pub mod metrics;

// Internal modules - used by Benchmark but not part of primary API
pub(crate) mod retry;
pub(crate) mod state;
pub(crate) mod worker;

// Supporting modules - available but not primary
pub mod datasets;
pub mod evaluators;
pub mod proxy;
pub mod runner;
