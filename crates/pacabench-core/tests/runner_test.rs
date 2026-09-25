//! Tests for the runner module.

use pacabench_core::config::AgentConfig;
use pacabench_core::runner::CommandRunner;
use pacabench_core::types::Case;
use tempfile::tempdir;

fn make_case() -> Case {
    Case {
        case_id: "1".into(),
        dataset_name: "ds".into(),
        input: "hello".into(),
        expected: None,
        history: vec![],
        metadata: Default::default(),
    }
}

#[tokio::test]
async fn command_runner_runs_and_returns_output() {
    let dir = tempdir().unwrap();
    let script = dir.path().join("agent.py");
    std::fs::write(
        &script,
        r#"
import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    data = json.loads(line)
    out = {"output": data.get("input", "") + "!", "error": None}
    print(json.dumps(out))
    sys.stdout.flush()
"#,
    )
    .unwrap();

    let cfg = AgentConfig {
        name: "echo".into(),
        command: format!("python {}", script.display()),
        setup: None,
        teardown: None,
        env: Default::default(),
    };
    let mut runner = CommandRunner::new(cfg, None, None, None);
    runner.start().await.unwrap();
    let res = runner.run_case(&make_case()).await.unwrap();
    assert_eq!(res.output.as_deref(), Some("hello!"));
    runner.stop().await.unwrap();
}

#[tokio::test]
async fn command_runner_respects_work_dir() {
    let dir = tempdir().unwrap();
    let work_dir = dir.path().to_path_buf();

    let script = work_dir.join("agent.py");
    std::fs::write(
        &script,
        r#"
import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    data = json.loads(line)
    out = {"output": "worked", "error": None}
    print(json.dumps(out))
    sys.stdout.flush()
"#,
    )
    .unwrap();

    let cfg = AgentConfig {
        name: "echo".into(),
        command: "python agent.py".into(),
        setup: None,
        teardown: None,
        env: Default::default(),
    };
    let mut runner = CommandRunner::new(cfg, None, None, Some(work_dir));
    runner.start().await.unwrap();
    let res = runner.run_case(&make_case()).await.unwrap();
    assert_eq!(res.output.as_deref(), Some("worked"));
    runner.stop().await.unwrap();
}

#[tokio::test]
async fn command_runner_flattens_metadata_to_top_level() {
    let dir = tempdir().unwrap();
    let script = dir.path().join("agent.py");
    std::fs::write(
        &script,
        r#"
import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    data = json.loads(line)
    choices = data.get("choices", {})
    out = {"output": f"got {len(choices)} choices", "error": None}
    print(json.dumps(out))
    sys.stdout.flush()
"#,
    )
    .unwrap();

    let cfg = AgentConfig {
        name: "echo".into(),
        command: format!("python {}", script.display()),
        setup: None,
        teardown: None,
        env: Default::default(),
    };

    let mut case = make_case();
    case.metadata.insert(
        "choices".to_string(),
        serde_json::json!({"A": "foo", "B": "bar", "C": "baz", "D": "qux"}),
    );

    let mut runner = CommandRunner::new(cfg, None, None, None);
    runner.start().await.unwrap();
    let res = runner.run_case(&case).await.unwrap();
    assert_eq!(
        res.output.as_deref(),
        Some("got 4 choices"),
        "metadata fields should be flattened to top level"
    );
    runner.stop().await.unwrap();
}

#[tokio::test]
async fn command_runner_proxy_url_passed_to_env() {
    let dir = tempdir().unwrap();
    let script = dir.path().join("agent.py");
    std::fs::write(
        &script,
        r#"
import sys, json, os
for line in sys.stdin:
    if not line.strip():
        continue
    base_url = os.environ.get("OPENAI_BASE_URL", "NOT_SET")
    out = {"output": base_url, "error": None}
    print(json.dumps(out))
    sys.stdout.flush()
"#,
    )
    .unwrap();

    let cfg = AgentConfig {
        name: "echo".into(),
        command: format!("python {}", script.display()),
        setup: None,
        teardown: None,
        env: Default::default(),
    };

    let proxy_url = "http://127.0.0.1:8080/v1".to_string();
    let mut runner = CommandRunner::new(cfg, Some(proxy_url), None, None);
    runner.start().await.unwrap();
    let res = runner.run_case(&make_case()).await.unwrap();
    assert_eq!(res.output.as_deref(), Some("http://127.0.0.1:8080/v1"));
    runner.stop().await.unwrap();
}

/// True if the process exists and is not a zombie awaiting reaping.
#[cfg(unix)]
fn process_alive(pid: i32) -> bool {
    // SAFETY: kill(2) with signal 0 only probes for existence.
    let exists = unsafe { libc::kill(pid, 0) == 0 };
    if !exists {
        return false;
    }
    // A zombie still answers kill(0); on Linux check the state in /proc.
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => !stat.contains(") Z"),
        Err(_) => exists,
    }
}

#[cfg(unix)]
#[tokio::test]
async fn command_runner_stop_kills_grandchildren() {
    let dir = tempdir().unwrap();
    let script = dir.path().join("agent.sh");
    // On the first case, spawn a long-lived grandchild and report its pid,
    // then keep the agent alive waiting for more input.
    std::fs::write(
        &script,
        r#"
read line
sleep 300 &
echo "{\"output\": \"$!\"}"
read line2
"#,
    )
    .unwrap();

    let cfg = AgentConfig {
        name: "spawner".into(),
        command: format!("sh {}", script.display()),
        setup: None,
        teardown: None,
        env: Default::default(),
    };
    let mut runner = CommandRunner::new(cfg, None, None, None);
    runner.start().await.unwrap();
    let res = runner.run_case(&make_case()).await.unwrap();
    let pid: i32 = res.output.expect("agent output").trim().parse().unwrap();
    assert!(process_alive(pid), "grandchild should be alive before stop");

    runner.stop().await.unwrap();

    // The process-group SIGKILL must take out the grandchild too; poll
    // briefly to allow signal delivery and reaping.
    let mut alive = true;
    for _ in 0..50 {
        if !process_alive(pid) {
            alive = false;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(!alive, "grandchild sleep (pid {pid}) survived runner stop");
}
