//! Runner process management: spawn agent commands and exchange JSONL over stdin/stdout.
//!
//! The `CommandRunner` manages the lifecycle of an agent subprocess, communicating
//! via JSONL over stdin/stdout. Each case is sent as a JSON object and the runner
//! waits for a response with `output` or `error` fields.
//!
//! # Thread Safety
//!
//! `CommandRunner` is NOT thread-safe internally. The caller (typically the orchestrator)
//! must ensure exclusive access when calling `run_case()`. This is by design - the
//! orchestrator manages concurrency at a higher level using semaphores and task spawning.

use crate::config::AgentConfig;
use crate::error::{PacabenchError, Result};
use crate::types::{Case, ErrorType};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;
use tracing::warn;

// ============================================================================
// RUNNER OUTPUT
// ============================================================================

/// Output from running a case through an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RunnerOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub error_type: ErrorType,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_traceback: Option<String>,
}

impl RunnerOutput {
    pub fn success(output: String, duration_ms: f64) -> Self {
        Self {
            output: Some(output),
            error: None,
            error_type: ErrorType::None,
            duration_ms,
            error_traceback: None,
        }
    }

    pub fn failure(error: String, error_type: ErrorType, duration_ms: f64) -> Self {
        Self {
            output: None,
            error: Some(error),
            error_type,
            duration_ms,
            error_traceback: None,
        }
    }

    pub fn is_success(&self) -> bool {
        self.output.is_some() && self.error.is_none()
    }
}

/// Manages an agent subprocess for running benchmark cases.
///
/// The runner spawns a subprocess using the configured command and communicates
/// via JSONL over stdin/stdout. It handles process lifecycle including setup,
/// teardown, and automatic restart if the process exits unexpectedly.
pub struct CommandRunner {
    config: AgentConfig,
    proxy_url: Option<String>,
    base_env: HashMap<String, String>,
    work_dir: Option<PathBuf>,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    stderr_task: Option<JoinHandle<()>>,
}

impl CommandRunner {
    /// Create a new runner for the given agent configuration.
    ///
    /// # Arguments
    /// * `config` - Agent configuration with command, env, setup/teardown
    /// * `proxy_url` - Optional proxy URL to set as OPENAI_BASE_URL
    /// * `base_env` - Base environment variables (defaults to current process env)
    /// * `work_dir` - Working directory for the subprocess
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
        }
    }

    /// Get the agent name.
    pub fn agent_name(&self) -> &str {
        &self.config.name
    }

    /// Check if the runner process is currently running.
    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }

    /// Start the runner subprocess.
    ///
    /// Runs the setup command (if configured), then spawns the main command.
    /// The subprocess communicates via JSONL over stdin/stdout.
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
            let status = setup_cmd.status().await.map_err(PacabenchError::Process)?;
            if !status.success() {
                return Err(anyhow!("setup command failed: {status}").into());
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

        let mut child = cmd.spawn().map_err(PacabenchError::Process)?;
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

    /// Stop the runner subprocess.
    ///
    /// Kills the subprocess, aborts the stderr reader task, and runs the
    /// teardown command (if configured).
    pub async fn stop(&mut self) -> Result<()> {
        self.kill_process();

        // Run teardown if provided
        if let Some(teardown) = &self.config.teardown {
            let mut teardown_cmd = Command::new("sh");
            teardown_cmd.arg("-c").arg(teardown);
            if let Some(dir) = &self.work_dir {
                if !dir.as_os_str().is_empty() {
                    teardown_cmd.current_dir(dir);
                }
            }
            let status = teardown_cmd
                .status()
                .await
                .map_err(PacabenchError::Process)?;
            if !status.success() {
                warn!(agent = %self.config.name, "teardown command failed: {}", status);
            }
        }
        Ok(())
    }

    /// Kill the subprocess without running teardown (used by Drop).
    fn kill_process(&mut self) {
        if let Some(ref mut child) = self.child {
            // start_kill() is synchronous and initiates the kill
            let _ = child.start_kill();
        }
        if let Some(handle) = self.stderr_task.take() {
            handle.abort();
        }
        self.child = None;
        self.stdin = None;
        self.stdout = None;
    }

    /// Execute a single case and return the runner output.
    ///
    /// If the runner process is not started, it will be started automatically.
    /// The case is sent as a JSONL line to stdin and we wait for a response
    /// containing `output` or `error` fields.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The runner process fails to start
    /// - stdin/stdout communication fails
    /// - The runner process exits unexpectedly
    pub async fn run_case(&mut self, case: &Case) -> Result<RunnerOutput> {
        // Auto-start if not running
        if self.child.is_none() {
            self.start().await?;
        }

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
                return Ok(RunnerOutput::failure(
                    "Process exited unexpectedly (EOF)".into(),
                    ErrorType::SystemFailure,
                    0.0,
                ));
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

/// RAII cleanup: kill the subprocess when the runner is dropped.
impl Drop for CommandRunner {
    fn drop(&mut self) {
        self.kill_process();
    }
}
