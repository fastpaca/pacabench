//! Core library for the Rust rewrite of PacaBench.
//!
//! This crate provides the building blocks for running LLM agent benchmarks:
//!
//! - [`config`]: Configuration loading and validation
//! - [`datasets`]: Dataset loaders (local, git, HuggingFace)
//! - [`evaluators`]: Result evaluation (F1, exact match, LLM judge)
//! - [`orchestrator`]: Main benchmark execution pipeline
//! - [`proxy`]: OpenAI-compatible proxy for metrics collection
//! - [`events`]: Event bus for observability and progress reporting
//! - [`state`]: Run state machine for lifecycle management
//! - [`error`]: Unified error types
//!
//! # Architecture
//!
//! The orchestrator spawns runner processes and a metrics proxy, executes
//! cases concurrently, and streams results via an event bus. The state
//! machine tracks run lifecycle for checkpointing and resume.

// Foundation modules (no internal dependencies)
pub mod state;
pub mod types;

// Error types (depends on state)
pub mod error;

// Core modules
pub mod config;
pub mod events;
pub mod pricing;

// Data loading
pub mod datasets;

// Execution
pub mod evaluators;
pub mod metrics;
pub mod orchestrator;
pub mod persistence;
pub mod proxy;
pub mod reporter;
pub mod run_manager;
pub mod runner;
