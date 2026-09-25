//! Retry policy for failed cases.

use crate::types::ErrorType;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Upper bound for a single backoff delay, before jitter is added.
const BACKOFF_CAP_MS: u64 = 10_000;

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
    pub fn should_retry(&self, attempt: u32, error_type: &ErrorType) -> bool {
        attempt < self.max_retries && error_type.is_retryable()
    }

    /// Backoff before retrying after the given attempt (1-based).
    ///
    /// Exponential (`base * 2^(attempt-1)`) with a cap, plus up to 25% random
    /// jitter so concurrent retries do not stampede in lockstep.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let backoff_ms = self.capped_exponential_ms(attempt);
        // Clock-derived jitter avoids pulling in a rand dependency.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::from(d.subsec_nanos()))
            .unwrap_or(0);
        let jitter_ms = nanos % (backoff_ms / 4 + 1);
        Duration::from_millis(backoff_ms + jitter_ms)
    }

    fn capped_exponential_ms(&self, attempt: u32) -> u64 {
        // Clamp the exponent so the shift cannot overflow.
        let exponent = attempt.saturating_sub(1).min(32);
        self.backoff_base_ms
            .saturating_mul(1u64 << exponent)
            .min(BACKOFF_CAP_MS)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_exponentially() {
        let policy = RetryPolicy::new(3, 100);
        assert_eq!(policy.capped_exponential_ms(1), 100);
        assert_eq!(policy.capped_exponential_ms(2), 200);
        assert_eq!(policy.capped_exponential_ms(3), 400);
        assert_eq!(policy.capped_exponential_ms(4), 800);
    }

    #[test]
    fn backoff_is_capped() {
        let policy = RetryPolicy::new(3, 100);
        assert_eq!(policy.capped_exponential_ms(30), BACKOFF_CAP_MS);
        assert_eq!(policy.capped_exponential_ms(u32::MAX), BACKOFF_CAP_MS);
    }

    #[test]
    fn backoff_duration_jitter_is_bounded() {
        let policy = RetryPolicy::new(3, 100);
        for _ in 0..100 {
            let ms = policy.backoff_duration(2).as_millis() as u64;
            assert!((200..=250).contains(&ms), "duration {ms}ms out of bounds");
        }
    }

    #[test]
    fn should_retry_respects_attempts_and_error_type() {
        let policy = RetryPolicy::new(2, 100);
        assert!(policy.should_retry(1, &ErrorType::SystemFailure));
        assert!(!policy.should_retry(2, &ErrorType::SystemFailure));
        assert!(!policy.should_retry(1, &ErrorType::TaskFailure));
        assert!(!policy.should_retry(1, &ErrorType::None));
    }
}
