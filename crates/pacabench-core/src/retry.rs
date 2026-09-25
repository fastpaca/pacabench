use crate::types::ErrorType;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Upper bound for one backoff delay, before jitter.
const BACKOFF_CAP_MS: u64 = 10_000;

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

    /// Whether a failed attempt should be run again.
    pub fn should_retry(&self, attempt: u32, error_type: &ErrorType) -> bool {
        attempt < self.max_retries && error_type.is_retryable()
    }

    /// Delay before the retry that follows `attempt` (1-based).
    ///
    /// Exponential (`base * 2^(attempt-1)`), capped, plus up to 25% jitter
    /// taken from the clock so concurrent retries do not line up.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let backoff_ms = self.capped_exponential_ms(attempt);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::from(d.subsec_nanos()))
            .unwrap_or(0);
        let jitter_ms = nanos % (backoff_ms / 4 + 1);
        Duration::from_millis(backoff_ms + jitter_ms)
    }

    fn capped_exponential_ms(&self, attempt: u32) -> u64 {
        let exponent = attempt.saturating_sub(1).min(32);
        self.backoff_base_ms
            .saturating_mul(1u64 << exponent)
            .min(BACKOFF_CAP_MS)
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
        let mut saw_jitter = false;
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_millis(5) {
            let ms = policy.backoff_duration(2).as_millis() as u64;
            assert!(
                (200..=250).contains(&ms),
                "duration {ms}ms outside 200..=250"
            );
            if ms > 200 {
                saw_jitter = true;
            }
        }
        assert!(saw_jitter, "expected clock jitter above the 200ms base");
    }

    #[test]
    fn should_retry_respects_attempts_and_error_type() {
        let policy = RetryPolicy::new(2, 100);
        assert!(policy.should_retry(1, &ErrorType::SystemFailure));
        assert!(!policy.should_retry(2, &ErrorType::SystemFailure));
        assert!(!policy.should_retry(1, &ErrorType::TaskFailure));
        assert!(!policy.should_retry(1, &ErrorType::FatalError));
        assert!(!policy.should_retry(1, &ErrorType::None));
    }
}
