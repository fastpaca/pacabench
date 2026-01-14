use super::{
    detect_dataset_file_format, prepare_case, read_json_array, resolve_path, DatasetContext,
    DatasetFileFormat, DatasetLoader,
};
use crate::config::DatasetConfig;
use crate::error::{PacabenchError, Result};
use crate::types::Case;
use async_trait::async_trait;
use futures_util::stream::{self, BoxStream, StreamExt, TryStreamExt};
use globwalk::GlobWalkerBuilder;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

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

    fn resolve_files(&self) -> Result<Vec<PathBuf>> {
        let source = &self.config.source;

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
                let walker = GlobWalkerBuilder::from_patterns(&p, &["*.jsonl", "*.json"])
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

        if let Some(s) = self.config.split.clone() {
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
                return Ok(filtered);
            }
        }

        Ok(files)
    }

    fn parse_case(
        &self,
        map: &serde_json::Map<String, Value>,
        file: &Path,
        idx: usize,
        input_key: &str,
        expected_key: &str,
    ) -> Option<Case> {
        prepare_case(
            map,
            &self.config.name,
            &format!("{}-{}", file.display(), idx),
            input_key,
            expected_key,
        )
    }
}

#[async_trait]
impl DatasetLoader for LocalDataset {
    async fn count_cases(&self, limit: Option<usize>) -> Result<usize> {
        let files = self.resolve_files()?;
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

        let mut count = 0usize;
        for file in files {
            match detect_dataset_file_format(&file).await? {
                DatasetFileFormat::JsonArray => {
                    let items = read_json_array(&file).await?;
                    for (idx, item) in items.into_iter().enumerate() {
                        if let Some(limit) = limit {
                            if count >= limit {
                                return Ok(count);
                            }
                        }
                        if let Value::Object(map) = item {
                            if self
                                .parse_case(&map, file.as_path(), idx, input_key, expected_key)
                                .is_some()
                            {
                                count += 1;
                            }
                        }
                    }
                }
                DatasetFileFormat::Jsonl => {
                    let f = File::open(&file)
                        .await
                        .map_err(PacabenchError::Persistence)?;
                    let reader = BufReader::new(f);
                    let mut lines = reader.lines();
                    let mut idx = 0usize;
                    while let Some(line) = lines
                        .next_line()
                        .await
                        .map_err(PacabenchError::Persistence)?
                    {
                        if let Some(limit) = limit {
                            if count >= limit {
                                return Ok(count);
                            }
                        }
                        let current_idx = idx;
                        idx += 1;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&line) {
                            if self
                                .parse_case(
                                    &map,
                                    file.as_path(),
                                    current_idx,
                                    input_key,
                                    expected_key,
                                )
                                .is_some()
                            {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    async fn stream_cases(&self, limit: Option<usize>) -> Result<BoxStream<'static, Result<Case>>> {
        let files = self.resolve_files()?;
        let dataset_name = self.config.name.clone();
        let input_key = self
            .config
            .input_map
            .get("input")
            .map(String::as_str)
            .unwrap_or("input")
            .to_string();
        let expected_key = self
            .config
            .input_map
            .get("expected")
            .map(String::as_str)
            .unwrap_or("expected")
            .to_string();

        let stream = stream::iter(files)
            .then(move |file| {
                let dataset_name = dataset_name.clone();
                let input_key = input_key.clone();
                let expected_key = expected_key.clone();
                async move {
                    match detect_dataset_file_format(&file).await? {
                        DatasetFileFormat::JsonArray => {
                            let items = read_json_array(&file).await?;
                            let cases = items
                                .into_iter()
                                .enumerate()
                                .filter_map(|(idx, item)| {
                                    if let Value::Object(map) = item {
                                        prepare_case(
                                            &map,
                                            &dataset_name,
                                            &format!("{}-{}", file.display(), idx),
                                            &input_key,
                                            &expected_key,
                                        )
                                        .map(Ok)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();
                            Ok::<_, PacabenchError>(stream::iter(cases).boxed())
                        }
                        DatasetFileFormat::Jsonl => {
                            let file_clone = file.clone();
                            let f = File::open(&file)
                                .await
                                .map_err(PacabenchError::Persistence)?;
                            let reader = BufReader::new(f);
                            let lines = tokio_stream::wrappers::LinesStream::new(reader.lines())
                                .enumerate()
                                .filter_map(move |(idx, line)| {
                                    let file = file_clone.clone();
                                    let dataset_name = dataset_name.clone();
                                    let input_key = input_key.clone();
                                    let expected_key = expected_key.clone();
                                    async move {
                                        match line {
                                            Ok(line) if !line.trim().is_empty() => {
                                                match serde_json::from_str::<Value>(&line) {
                                                    Ok(Value::Object(map)) => prepare_case(
                                                        &map,
                                                        &dataset_name,
                                                        &format!("{}-{}", file.display(), idx),
                                                        &input_key,
                                                        &expected_key,
                                                    )
                                                    .map(Ok),
                                                    _ => None,
                                                }
                                            }
                                            Ok(_) => None,
                                            Err(e) => Some(Err(PacabenchError::Internal(e.into()))),
                                        }
                                    }
                                });

                            Ok::<_, PacabenchError>(lines.boxed())
                        }
                    }
                }
            })
            .try_flatten()
            .take(limit.unwrap_or(usize::MAX));

        Ok(Box::pin(stream))
    }
}
