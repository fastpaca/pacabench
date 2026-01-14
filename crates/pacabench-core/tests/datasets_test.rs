//! Tests for the datasets module.

use futures_util::TryStreamExt;
use git2::Repository;
use pacabench_core::config::DatasetConfig;
use pacabench_core::datasets::{
    DatasetContext, DatasetLoader, GitDataset, HuggingFaceDataset, LocalDataset,
};
use serde_json::json;
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

#[tokio::test]
async fn local_dataset_loads_records() {
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
    let cases: Vec<_> = loader
        .stream_cases(None)
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].input, "hi");
    assert_eq!(cases[1].case_id, "2");
}

#[tokio::test]
async fn local_dataset_loads_json_array() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let file_path = data_dir.join("sample.json");
    std::fs::write(
        &file_path,
        json!([
            {"case_id": "1", "input": "hi", "expected": "hi"},
            {"case_id": "2", "input": "bye", "expected": "bye"}
        ])
        .to_string(),
    )
    .unwrap();

    let cfg = DatasetConfig {
        name: "test".into(),
        source: file_path.to_string_lossy().to_string(),
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
    let cases: Vec<_> = loader
        .stream_cases(None)
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].case_id, "1");
}

#[tokio::test]
async fn local_dataset_honors_split_file() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let train = data_dir.join("train.jsonl");
    let test = data_dir.join("test.jsonl");
    std::fs::write(
        &train,
        json!({"case_id": "train1", "input": "t", "expected": "t"}).to_string() + "\n",
    )
    .unwrap();
    std::fs::write(
        &test,
        json!({"case_id": "test1", "input": "x", "expected": "x"}).to_string() + "\n",
    )
    .unwrap();

    let cfg = DatasetConfig {
        name: "split".into(),
        source: data_dir.to_string_lossy().to_string(),
        split: Some("test".into()),
        prepare: None,
        input_map: Default::default(),
        evaluator: None,
    };
    let ctx = DatasetContext {
        root_dir: dir.path().to_path_buf(),
        cache_dir: dir.path().join("cache"),
    };
    let loader = LocalDataset::new(cfg, ctx);
    let cases: Vec<_> = loader
        .stream_cases(None)
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].case_id, "test1");
}

fn commit_repo(repo_path: &Path) {
    let repo = Repository::init(repo_path).unwrap();
    let mut index = repo.index().unwrap();
    index
        .add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)
        .unwrap();
    let tree_id = index.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    let sig = git2::Signature::now("tester", "test@example.com").unwrap();
    let head = repo.head().ok().and_then(|h| h.target());
    let parent = head.and_then(|oid| repo.find_commit(oid).ok());
    match parent {
        Some(p) => {
            repo.commit(Some("HEAD"), &sig, &sig, "update", &tree, &[&p])
                .unwrap();
        }
        None => {
            repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[])
                .unwrap();
        }
    }
}

#[tokio::test]
async fn git_dataset_runs_prepare_with_env() {
    let dir = tempdir().unwrap();
    let repo_src = dir.path().join("remote");
    std::fs::create_dir_all(&repo_src).unwrap();
    std::fs::write(
        repo_src.join("train.jsonl"),
        json!({"case_id": "g1", "input": "g", "expected": "g"}).to_string() + "\n",
    )
    .unwrap();
    commit_repo(&repo_src);

    let prepare_script = dir.path().join("prepare.sh");
    std::fs::write(
        &prepare_script,
        r#"#!/bin/sh
echo "$PACABENCH_DATASET_PATH" > prepare_marker.txt
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&prepare_script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&prepare_script, perms).unwrap();
    }

    let cfg = DatasetConfig {
        name: "gitds".into(),
        source: format!("git:{}", repo_src.to_string_lossy()),
        split: Some("train".into()),
        prepare: Some(prepare_script.to_string_lossy().to_string()),
        input_map: Default::default(),
        evaluator: None,
    };
    let ctx = DatasetContext {
        root_dir: dir.path().to_path_buf(),
        cache_dir: dir.path().join("cache"),
    };
    let loader = GitDataset::new(cfg, ctx.clone());
    let cases: Vec<_> = loader
        .stream_cases(None)
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].case_id, "g1");

    let marker = ctx.root_dir.join("prepare_marker.txt");
    assert!(marker.exists(), "prepare script should run in root dir");
    let contents = std::fs::read_to_string(marker).unwrap();
    assert!(
        contents.contains("repos"),
        "prepare should receive dataset path env"
    );
}

#[tokio::test]
async fn huggingface_loader_handles_local_split() {
    let dir = tempdir().unwrap();
    let hf_dir = dir.path().join("hfdata");
    std::fs::create_dir_all(&hf_dir).unwrap();
    std::fs::write(
        hf_dir.join("train.jsonl"),
        json!({"case_id": "h1", "input": "a", "expected": "a"}).to_string() + "\n",
    )
    .unwrap();
    std::fs::write(
        hf_dir.join("dev.jsonl"),
        json!({"case_id": "h2", "input": "b", "expected": "b"}).to_string() + "\n",
    )
    .unwrap();

    let cfg = DatasetConfig {
        name: "hf".into(),
        source: "huggingface:hfdata".into(),
        split: Some("dev".into()),
        prepare: None,
        input_map: Default::default(),
        evaluator: None,
    };
    let ctx = DatasetContext {
        root_dir: dir.path().to_path_buf(),
        cache_dir: dir.path().join("cache"),
    };
    let loader = HuggingFaceDataset::new(cfg, ctx);
    let cases: Vec<_> = loader
        .stream_cases(None)
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].case_id, "h2");
}

#[tokio::test]
async fn huggingface_loader_handles_json_array_split() {
    let dir = tempdir().unwrap();
    let hf_dir = dir.path().join("hfdata");
    std::fs::create_dir_all(&hf_dir).unwrap();
    std::fs::write(
        hf_dir.join("longmemeval_oracle.json"),
        json!([
            {"case_id": "h1", "input": "a", "expected": "a"},
            {"case_id": "h2", "input": "b", "expected": "b"}
        ])
        .to_string(),
    )
    .unwrap();

    let cfg = DatasetConfig {
        name: "hf".into(),
        source: "huggingface:hfdata".into(),
        split: Some("longmemeval_oracle".into()),
        prepare: None,
        input_map: Default::default(),
        evaluator: None,
    };
    let ctx = DatasetContext {
        root_dir: dir.path().to_path_buf(),
        cache_dir: dir.path().join("cache"),
    };
    let loader = HuggingFaceDataset::new(cfg, ctx);
    let cases: Vec<_> = loader
        .stream_cases(None)
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].case_id, "h1");
}
