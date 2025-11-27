//! Tests for the datasets module.

use pacabench_core::config::DatasetConfig;
use pacabench_core::datasets::{DatasetContext, DatasetLoader, LocalDataset};
use serde_json::json;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn local_dataset_loads_records() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let file_path = data_dir.join("sample.jsonl");
    let mut file = std::fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        "{}",
        json!({"case_id": "1", "input": "hi", "expected": "hi"})
    )
    .unwrap();
    writeln!(
        file,
        "{}",
        json!({"case_id": "2", "input": "bye", "expected": "bye", "metadata": {"foo": "bar"}})
    )
    .unwrap();

    let cfg = DatasetConfig {
        name: "test".into(),
        source: data_dir.to_string_lossy().to_string(),
        split: None,
        prepare: None,
        input_map: Default::default(),
        evaluator: None,
    };
    let ctx = DatasetContext {
        root_dir: dir.path().to_path_buf(),
        cache_dir: dir.path().join("cache"),
    };
    let loader = LocalDataset::new(cfg, ctx);
    let cases = loader.load(None).unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].input, "hi");
    assert_eq!(cases[1].case_id, "2");
}
