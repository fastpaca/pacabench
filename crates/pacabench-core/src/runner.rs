//! Runner process management: spawn agent commands and exchange JSONL over stdin/stdout.

use crate::config::AgentConfig;
use crate::types::{Case, ErrorType, RunnerOutput};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::warn;

pub struct CommandRunner {
    config: AgentConfig,
    proxy_url: Option<String>,
    base_env: HashMap<String, String>,
    work_dir: Option<PathBuf>,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    stderr_task: Option<JoinHandle<()>>,
    lock: Mutex<()>,
}

impl CommandRunner {
    pub fn new(
        config: AgentConfig,
        proxy_url: Option<String>,
        base_env: Option<HashMap<String, String>>,
        work_dir: Option<PathBuf>,
    ) -> Self {
        let env = base_env.unwrap_or_else(|| std::env::vars().collect());
        Self {
            config,
            proxy_url,
            base_env: env,
            work_dir,
            child: None,
            stdin: None,
            stdout: None,
            stderr_task: None,
            lock: Mutex::new(()),
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        // Run setup if provided (blocking).
        if let Some(setup) = &self.config.setup {
            let mut setup_cmd = Command::new("sh");
            setup_cmd.arg("-c").arg(setup);
            if let Some(dir) = &self.work_dir {
                if !dir.as_os_str().is_empty() {
                    setup_cmd.current_dir(dir);
                }
            }
            let status = setup_cmd.status().await?;
            if !status.success() {
                return Err(anyhow!("setup command failed: {status}"));
            }
        }

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&self.config.command);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(dir) = &self.work_dir {
            if !dir.as_os_str().is_empty() {
                cmd.current_dir(dir);
            }
        }

        // Build environment.
        let mut env = self.base_env.clone();
        env.extend(self.config.env.clone());
        if let Some(url) = &self.proxy_url {
            env.insert("OPENAI_BASE_URL".to_string(), url.clone());
        }
        cmd.envs(env);

        let mut child = cmd.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("child missing stdout"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("child missing stdin"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("child missing stderr"))?;

        let stderr_reader = BufReader::new(stderr);
        let agent_name = self.config.name.clone();
        let stderr_task = tokio::spawn(async move {
            let mut lines = stderr_reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if !line.trim().is_empty() {
                    warn!(agent = %agent_name, "[stderr] {}", line);
                }
            }
        });

        self.child = Some(child);
        self.stdin = Some(stdin);
        self.stdout = Some(BufReader::new(stdout));
        self.stderr_task = Some(stderr_task);
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<()> {
        if let Some(child) = self.child.as_mut() {
            child.kill().await.ok();
            child.wait().await.ok();
        }
        if let Some(handle) = self.stderr_task.take() {
            handle.abort();
        }
        self.child = None;
        self.stdin = None;
        self.stdout = None;

        // Run teardown if provided
        if let Some(teardown) = &self.config.teardown {
            let mut teardown_cmd = Command::new("sh");
            teardown_cmd.arg("-c").arg(teardown);
            if let Some(dir) = &self.work_dir {
                if !dir.as_os_str().is_empty() {
                    teardown_cmd.current_dir(dir);
                }
            }
            let status = teardown_cmd.status().await?;
            if !status.success() {
                warn!(agent = %self.config.name, "teardown command failed: {}", status);
            }
        }
        Ok(())
    }

    pub async fn run_case(&mut self, case: &Case) -> Result<RunnerOutput> {
        let mut guard = self.lock.lock().await;
        if self.child.is_none() {
            drop(guard);
            self.start().await?;
            guard = self.lock.lock().await;
        }
        let _guard = guard;

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("runner stdin unavailable"))?;
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("runner stdout unavailable"))?;

        let mut payload = serde_json::Map::new();
        payload.insert("case_id".into(), Value::String(case.case_id.clone()));
        payload.insert(
            "dataset_name".into(),
            Value::String(case.dataset_name.clone()),
        );
        payload.insert("input".into(), Value::String(case.input.clone()));
        payload.insert("history".into(), Value::Array(case.history.clone()));
        // Flatten metadata fields to top level for agent compatibility
        // (agents expect fields like "choices" at root, not nested in metadata)
        for (k, v) in &case.metadata {
            payload.insert(k.clone(), v.clone());
        }
        payload.insert(
            "metadata".into(),
            Value::Object(case.metadata.clone().into_iter().collect()),
        );

        let line = serde_json::to_string(&payload)? + "\n";
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;

        loop {
            let mut buf = String::new();
            let n = stdout.read_line(&mut buf).await?;
            if n == 0 {
                return Ok(RunnerOutput {
                    output: None,
                    error: Some("Process exited unexpectedly (EOF)".into()),
                    metrics: None,
                    duration_ms: 0.0,
                    error_type: ErrorType::SystemFailure,
                    error_traceback: None,
                    retry_count: 0,
                });
            }
            if buf.trim().is_empty() {
                continue;
            }
            let val: Value = match serde_json::from_str(&buf) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Value::Object(map) = val {
                if map.get("output").is_some() || map.get("error").is_some() {
                    let mut ro: RunnerOutput = serde_json::from_value(Value::Object(map.clone()))?;
                    if ro.metrics.is_none() {
                        // try to map metrics object if present
                        if let Some(Value::Object(obj)) = map.get("metrics") {
                            ro.metrics = Some(obj.clone().into_iter().collect());
                        }
                    }
                    // If no error_type but error present, mark as system failure.
                    if ro.error_type == ErrorType::None && ro.error.is_some() {
                        ro.error_type = ErrorType::SystemFailure;
                    }
                    return Ok(ro);
                }
            }
        }
    }
}
