//! PacaBench Core - LLM agent benchmarking library.
//!
//! # Quick Start
//!
//! ```no_run
//! use pacabench_core::{Benchmark, Config, Event};
//! use pacabench_core::config::ConfigOverrides;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load from config file
//!     let config = Config::from_file("pacabench.yaml", ConfigOverrides::default())?;
//!     let bench = Benchmark::new(config);
//!
//!     // Subscribe to events
//!     let mut events = bench.subscribe();
//!     tokio::spawn(async move {
//!         while let Ok(event) = events.recv().await {
//!             match event {
//!                 Event::CaseCompleted { passed, .. } => println!("Case: {}", if passed { "✓" } else { "✗" }),
//!                 Event::RunCompleted { stats, .. } => println!("Done: {:.1}% accuracy", stats.accuracy * 100.0),
//!                 _ => {}
//!             }
//!         }
//!     });
//!
//!     // Run
//!     let result = bench.run(None, None).await?;
//!     Ok(())
//! }
//! ```
//!
pub mod benchmark;
pub use benchmark::{Benchmark, RunResult};

pub mod config;
pub use config::Config;

pub mod types;
pub use types::{
    AggregatedMetrics, Case, CaseKey, CaseResult, Command, ErrorType, EvaluationResult, Event,
    JudgeMetrics, LlmMetrics, RunStatus,
};

pub use runner::RunnerOutput;

pub mod error;
pub mod metrics;
pub mod persistence;
pub mod stats;
pub use stats::RunStats;

pub(crate) mod retry;
pub(crate) mod state;
pub(crate) mod utils;
pub(crate) mod worker;

pub mod datasets;
pub mod evaluators;
pub mod proxy;
pub mod runner;
