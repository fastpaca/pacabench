//! Retry policy for failed cases.

use crate::types::ErrorType;
use std::time::Duration;

/// Policy for retrying failed cases.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff_base_ms: u64,
}

impl RetryPolicy {
    pub fn new(max_retries: u32, backoff_base_ms: u64) -> Self {
        Self {
            max_retries,
            backoff_base_ms,
        }
    }

    /// Check if a case should be retried given its attempt count and error type.
    #[allow(dead_code)]
    pub fn should_retry(&self, attempt: u32, error_type: &ErrorType) -> bool {
        attempt < self.max_retries && error_type.is_retryable()
    }

    /// Calculate backoff duration for a given attempt.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        Duration::from_millis(self.backoff_base_ms * (attempt as u64))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff_base_ms: 100,
        }
    }
}
