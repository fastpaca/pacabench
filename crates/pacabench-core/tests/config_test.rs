//! Tests for the config module.

use pacabench_core::config::{Config, ConfigOverrides};
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

    let err = Config::from_file(file.path(), ConfigOverrides::default()).unwrap_err();
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

    let cfg = Config::from_file(file.path(), ConfigOverrides::default()).unwrap();
    assert_eq!(cfg.global.concurrency, 4);
    assert!(cfg.global.proxy.enabled);
    assert_eq!(cfg.output.directory, "./runs");
}
