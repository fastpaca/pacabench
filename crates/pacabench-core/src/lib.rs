//! Core library for the Rust rewrite of PacaBench.
//!
//! This crate will host shared types, config loading, orchestration logic,
//! and other reusable building blocks for the CLI and any embedders.

pub mod config;
pub mod datasets;
pub mod evaluators;
pub mod metrics;
pub mod orchestrator;
pub mod persistence;
pub mod pricing;
pub mod proxy;
pub mod run_manager;
pub mod runner;
pub mod types;
