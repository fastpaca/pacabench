//! Tests for the config module.

use pacabench_core::config::load_config;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn load_config_requires_agents_and_datasets() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        r#"
name: sample
version: "0.1.0"
agents: []
datasets: []
"#
    )
    .unwrap();

    let err = load_config(file.path()).unwrap_err();
    assert!(format!("{err}").contains("at least one agent"));
}

#[test]
fn load_config_applies_defaults() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        r#"
name: defaults
agents:
  - name: a1
    command: "echo hi"
datasets:
  - name: ds1
    source: "data.jsonl"
"#
    )
    .unwrap();

    let cfg = load_config(file.path()).unwrap();
    assert_eq!(cfg.config.concurrency, 4);
    assert!(cfg.config.proxy.enabled);
    assert_eq!(cfg.output.directory, "./runs");
}
