//! Run state machine for lifecycle management.
//!
//! Provides explicit, type-safe state transitions for benchmark runs.
//! All state is stored atomically for lock-free concurrent access.

use crate::error::PacabenchError;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// The possible states of a benchmark run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RunStateKind {
    /// Run is being initialized.
    Initializing = 0,
    /// Loading datasets.
    Loading = 1,
    /// Executing cases.
    Running = 2,
    /// Run was paused (e.g., for checkpointing).
    Paused = 3,
    /// Run completed successfully.
    Completed = 4,
    /// Run failed due to errors.
    Failed = 5,
}

impl RunStateKind {
    fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Initializing,
            1 => Self::Loading,
            2 => Self::Running,
            3 => Self::Paused,
            4 => Self::Completed,
            5 => Self::Failed,
            _ => Self::Failed,
        }
    }

    /// Check if transition to `to` is valid from this state.
    pub fn can_transition_to(self, to: Self) -> bool {
        use RunStateKind::*;
        matches!(
            (self, to),
            // Normal flow
            (Initializing, Loading)
                | (Loading, Running)
                | (Running, Completed)
                // Pause/resume
                | (Running, Paused)
                | (Paused, Running)
                // Failures from any non-terminal state
                | (Initializing, Failed)
                | (Loading, Failed)
                | (Running, Failed)
                | (Paused, Failed)
        )
    }
}

/// Atomic state machine for tracking run lifecycle and progress.
///
/// All operations are lock-free using atomic primitives.
#[derive(Debug)]
pub struct RunStateMachine {
    state: AtomicU8,
    total: AtomicU64,
    completed: AtomicU64,
    passed: AtomicU64,
    failed: AtomicU64,
    system_errors: AtomicU64,
}

impl RunStateMachine {
    /// Create a new state machine in `Initializing` state.
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(RunStateKind::Initializing as u8),
            total: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            passed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            system_errors: AtomicU64::new(0),
        }
    }

    /// Create a state machine with known totals (for resume).
    pub fn with_totals(total: u64, completed: u64, passed: u64) -> Self {
        Self {
            state: AtomicU8::new(RunStateKind::Initializing as u8),
            total: AtomicU64::new(total),
            completed: AtomicU64::new(completed),
            passed: AtomicU64::new(passed),
            failed: AtomicU64::new(completed.saturating_sub(passed)),
            system_errors: AtomicU64::new(0),
        }
    }

    /// Get the current state.
    pub fn state(&self) -> RunStateKind {
        RunStateKind::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Attempt to transition to a new state.
    ///
    /// Returns the previous state on success, or an error if the transition is invalid.
    pub fn transition(&self, to: RunStateKind) -> Result<RunStateKind, PacabenchError> {
        let current = self.state();
        if !current.can_transition_to(to) {
            return Err(PacabenchError::InvalidTransition {
                from: current,
                to,
            });
        }
        self.state.store(to as u8, Ordering::Release);
        Ok(current)
    }

    /// Set the total number of cases.
    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Release);
    }

    /// Get the total number of cases.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Acquire)
    }

    /// Get the number of completed cases.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Get the number of passed cases.
    pub fn passed(&self) -> u64 {
        self.passed.load(Ordering::Acquire)
    }

    /// Get the number of failed cases.
    pub fn failed(&self) -> u64 {
        self.failed.load(Ordering::Acquire)
    }

    /// Get the number of system errors.
    pub fn system_errors(&self) -> u64 {
        self.system_errors.load(Ordering::Acquire)
    }

    /// Record a completed case.
    pub fn record_completed(&self, passed: bool, is_system_error: bool) {
        self.completed.fetch_add(1, Ordering::AcqRel);
        if passed {
            self.passed.fetch_add(1, Ordering::AcqRel);
        } else {
            self.failed.fetch_add(1, Ordering::AcqRel);
        }
        if is_system_error {
            self.system_errors.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Get current progress as a fraction (0.0 - 1.0).
    pub fn progress(&self) -> f64 {
        let total = self.total.load(Ordering::Acquire);
        if total == 0 {
            return 0.0;
        }
        let completed = self.completed.load(Ordering::Acquire);
        completed as f64 / total as f64
    }

    /// Get current error ratio.
    pub fn error_ratio(&self) -> f64 {
        let completed = self.completed.load(Ordering::Acquire);
        if completed == 0 {
            return 0.0;
        }
        let errors = self.system_errors.load(Ordering::Acquire);
        errors as f64 / completed as f64
    }

    /// Create a snapshot for checkpointing.
    pub fn snapshot(&self) -> RunSnapshot {
        RunSnapshot {
            state: self.state(),
            total: self.total.load(Ordering::Acquire),
            completed: self.completed.load(Ordering::Acquire),
            passed: self.passed.load(Ordering::Acquire),
            failed: self.failed.load(Ordering::Acquire),
            system_errors: self.system_errors.load(Ordering::Acquire),
        }
    }
}

impl Default for RunStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of run state (for checkpointing).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSnapshot {
    pub state: RunStateKind,
    pub total: u64,
    pub completed: u64,
    pub passed: u64,
    pub failed: u64,
    pub system_errors: u64,
}

