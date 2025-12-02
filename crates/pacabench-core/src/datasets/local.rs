use super::{prepare_case, resolve_path, DatasetContext, DatasetLoader};
use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use globwalk::GlobWalkerBuilder;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub struct LocalDataset {
    config: DatasetConfig,
    root: PathBuf,
}

impl LocalDataset {
    pub fn new(config: DatasetConfig, ctx: DatasetContext) -> Self {
        Self {
            config,
            root: ctx.root_dir,
        }
    }
}

impl DatasetLoader for LocalDataset {
    fn load(&self, limit: Option<usize>) -> Result<Vec<Case>> {
        let source = &self.config.source;
        let split = self.config.split.clone();
        let input_key = self
            .config
            .input_map
            .get("input")
            .map(String::as_str)
            .unwrap_or("input");
        let expected_key = self
            .config
            .input_map
            .get("expected")
            .map(String::as_str)
            .unwrap_or("expected");

        let mut files: Vec<PathBuf> = Vec::new();
        if source.contains('*') {
            let pattern = resolve_path(source, &self.root);
            let walker = GlobWalkerBuilder::from_patterns(
                pattern
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| self.root.clone()),
                &[pattern
                    .file_name()
                    .map(|os| os.to_string_lossy().to_string())
                    .unwrap_or_else(|| "*.jsonl".to_string())],
            )
            .build()
            .map_err(|e| PacabenchError::Internal(e.into()))?;
            for entry in walker.into_iter().filter_map(|e| e.ok()) {
                if entry.path().is_file() {
                    files.push(entry.path().to_path_buf());
                }
            }
        } else {
            let p = resolve_path(source, &self.root);
            if p.is_dir() {
                let walker = GlobWalkerBuilder::from_patterns(&p, &["*.jsonl"])
                    .build()
                    .map_err(|e| PacabenchError::Internal(e.into()))?;
                for entry in walker.into_iter().filter_map(|e| e.ok()) {
                    if entry.path().is_file() {
                        files.push(entry.path().to_path_buf());
                    }
                }
            } else {
                files.push(p);
            }
        }

        if let Some(s) = split {
            let filtered: Vec<PathBuf> = files
                .iter()
                .filter(|p| {
                    p.file_stem()
                        .and_then(|f| f.to_str())
                        .map(|stem| stem == s)
                        .unwrap_or(false)
                        || p.file_name()
                            .and_then(|f| f.to_str())
                            .map(|name| name.contains(&s))
                            .unwrap_or(false)
                })
                .cloned()
                .collect();
            if !filtered.is_empty() {
                files = filtered;
            }
        }

        let mut cases = Vec::new();
        let mut count = 0usize;
        for file in files {
            let f = File::open(&file)?;
            let reader = BufReader::new(f);
            for (idx, line) in reader.lines().enumerate() {
                if let Some(limit) = limit {
                    if count >= limit {
                        return Ok(cases);
                    }
                }
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&line) {
                    if let Some(case) = prepare_case(
                        &map,
                        &self.config.name,
                        &format!("{}-{}", file.display(), idx),
                        input_key,
                        expected_key,
                    ) {
                        cases.push(case);
                        count += 1;
                    }
                }
            }
        }

        Ok(cases)
    }
}
