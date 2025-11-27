//! Tests for the config module.

use pacabench_core::config::interpolate_env_vars;

#[test]
fn test_env_interpolation_basic() {
    std::env::set_var("TEST_VAR_123", "hello");
    let result = interpolate_env_vars("value: ${TEST_VAR_123}");
    assert_eq!(result, "value: hello");
    std::env::remove_var("TEST_VAR_123");
}

#[test]
fn test_env_interpolation_with_default() {
    std::env::remove_var("NONEXISTENT_VAR_XYZ");
    let result = interpolate_env_vars("value: ${NONEXISTENT_VAR_XYZ:-default_value}");
    assert_eq!(result, "value: default_value");
}

#[test]
fn test_env_interpolation_missing_no_default() {
    std::env::remove_var("NONEXISTENT_VAR_ABC");
    let result = interpolate_env_vars("value: ${NONEXISTENT_VAR_ABC}");
    assert_eq!(result, "value: ");
}

#[test]
fn test_env_interpolation_multiple() {
    std::env::set_var("VAR_A", "alpha");
    std::env::set_var("VAR_B", "beta");
    let result = interpolate_env_vars("a: ${VAR_A}, b: ${VAR_B}");
    assert_eq!(result, "a: alpha, b: beta");
    std::env::remove_var("VAR_A");
    std::env::remove_var("VAR_B");
}
