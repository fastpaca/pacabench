//! Tests for the pricing module.

use pacabench_core::pricing::{calculate_cost, get_pricing};

#[test]
fn test_gpt4o_mini_pricing() {
    let pricing = get_pricing("gpt-4o-mini").unwrap();
    assert!((pricing.input_per_1m - 0.15).abs() < 0.001);
    assert!((pricing.output_per_1m - 0.60).abs() < 0.001);
}

#[test]
fn test_cost_calculation() {
    let cost = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000, 0);
    assert!((cost - 0.75).abs() < 0.001);
}

#[test]
fn test_version_matching() {
    // Exact version should match
    let pricing = get_pricing("gpt-4o-mini-2024-07-18").unwrap();
    assert!((pricing.input_per_1m - 0.15).abs() < 0.001);
}

#[test]
fn test_unknown_model() {
    let pricing = get_pricing("unknown-model");
    assert!(pricing.is_none());
    let cost = calculate_cost("unknown-model", 1000, 1000, 0);
    assert_eq!(cost, 0.0);
}
