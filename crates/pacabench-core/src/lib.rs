//! PacaBench Core - LLM agent benchmarking library.
//!
//! # Quick Start
//!
//! ```ignore
//! use pacabench_core::{Benchmark, Config, Event};
//! use pacabench_core::config::ConfigOverrides;
//!
//! // Load from config file
//! let config = Config::from_file("pacabench.yaml", ConfigOverrides::default())?;
//! let bench = Benchmark::new(config);
//!
//! // Subscribe to events
//! let events = bench.subscribe();
//! tokio::spawn(async move {
//!     while let Ok(event) = events.recv() {
//!         match event {
//!             Event::CaseCompleted { passed, .. } => println!("Case: {}", if passed { "✓" } else { "✗" }),
//!             Event::RunCompleted { metrics, .. } => println!("Done: {:.1}% accuracy", metrics.accuracy * 100.0),
//!             _ => {}
//!         }
//!     }
//! });
//!
//! // Run
//! let result = bench.run(None, None).await?;
//! ```
//!
//! # Public API
//!
//! - [`Config`] - Configuration (load with `Config::from_file()`)
//! - [`Benchmark`] - Main entry point (`Benchmark::new(config)`)
//! - [`Event`] - Events emitted during execution
//! - [`Command`] - Commands to control execution (stop, abort)
//! - [`RunResult`](benchmark::RunResult) - Result of a benchmark run

// Public API
pub mod benchmark;
pub use benchmark::{Benchmark, RunResult};

pub mod config;
pub use config::Config;

pub mod protocol;
pub use protocol::{Command, Event};

pub mod types;
pub use types::{
    AggregatedMetrics, Case, CaseKey, CaseResult, ErrorType, EvaluationResult, JudgeMetrics,
    LlmMetrics, RunStatus, RunnerOutput,
};

pub mod error;
pub mod metrics;
pub mod persistence;

// Internal modules
pub(crate) mod retry;
pub(crate) mod state;
pub(crate) mod worker;

// Supporting modules
pub mod datasets;
pub mod evaluators;
pub mod proxy;
pub mod runner;
