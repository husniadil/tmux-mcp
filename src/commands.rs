//! Command execution tracking with markers for tmux-mcp.
//!
//! This module provides the `CommandTracker` struct that manages command execution
//! in tmux panes, using special markers to track command start/completion and exit codes.

use std::collections::HashMap;
#[cfg(test)]
use std::ffi::OsString;
use std::sync::Arc;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::Deserialize;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::errors::Result;
use crate::tmux;
use crate::types::{CommandExecution, CommandStatus, ShellType};

/// Prefix for the start marker, followed by command id.
pub const START_MARKER_PREFIX: &str = "TMUX_MCP_START_";

/// Prefix for the end marker, followed by command id and exit code.
pub const END_MARKER_PREFIX: &str = "TMUX_MCP_DONE_";

#[cfg(test)]
struct EnvVarGuard {
    key: &'static str,
    prev: Option<OsString>,
}

#[cfg(test)]
impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, prev }
    }
}

#[cfg(test)]
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            std::env::set_var(self.key, prev);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

/// Tracking configuration for command capture retries.
#[derive(Debug, Clone, Deserialize)]
pub struct TrackingConfig {
    #[serde(default = "default_capture_initial_lines")]
    pub capture_initial_lines: u32,
    #[serde(default = "default_capture_max_lines")]
    pub capture_max_lines: u32,
    #[serde(default = "default_capture_backoff_factor")]
    pub capture_backoff_factor: u32,
    #[serde(default = "default_completed_retention_minutes")]
    pub completed_retention_minutes: u64,
    #[serde(default = "default_completed_max_entries")]
    pub completed_max_entries: u32,
    #[serde(default = "default_tracking_deadline_seconds")]
    pub tracking_deadline_seconds: u64,
}

fn default_capture_initial_lines() -> u32 {
    1000
}

fn default_capture_max_lines() -> u32 {
    16_000
}

fn default_capture_backoff_factor() -> u32 {
    2
}

fn default_completed_retention_minutes() -> u64 {
    240
}

fn default_completed_max_entries() -> u32 {
    1000
}

/// How long a tracked command whose START marker is no longer reachable in the
/// pane history (e.g. it scrolled past `capture_max_lines` under heavy output)
/// stays Pending before being declared expired. Generous on purpose: while the
/// anchor is lost we cannot tell "still running" from "gone", so we err toward
/// waiting. Raise it for workloads that emit very large output over long runs.
fn default_tracking_deadline_seconds() -> u64 {
    600
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            capture_initial_lines: default_capture_initial_lines(),
            capture_max_lines: default_capture_max_lines(),
            capture_backoff_factor: default_capture_backoff_factor(),
            completed_retention_minutes: default_completed_retention_minutes(),
            completed_max_entries: default_completed_max_entries(),
            tracking_deadline_seconds: default_tracking_deadline_seconds(),
        }
    }
}

/// Tracks active and recently completed commands across tmux panes.
#[derive(Debug)]
pub struct CommandTracker {
    active_commands: Arc<RwLock<HashMap<String, CommandExecution>>>,
    shell_type: ShellType,
    tracking: TrackingConfig,
}

impl CommandTracker {
    /// Create a new CommandTracker for the given shell type.
    pub fn new(shell_type: ShellType) -> Self {
        Self::with_tracking(shell_type, TrackingConfig::default())
    }

    /// Create a new CommandTracker with custom tracking configuration.
    pub fn with_tracking(shell_type: ShellType, tracking: TrackingConfig) -> Self {
        Self {
            active_commands: Arc::new(RwLock::new(HashMap::new())),
            shell_type,
            tracking,
        }
    }

    /// Execute a command in a tmux pane with optional tracking markers.
    ///
    /// Returns the command ID that can be used to check status.
    pub async fn execute_command(
        &self,
        pane_id: &str,
        command: &str,
        raw_mode: bool,
        no_enter: bool,
        delay_ms: Option<u64>,
        socket: Option<String>,
    ) -> Result<String> {
        let command_id = Uuid::new_v4().to_string();
        let resolved_socket = tmux::resolve_socket(socket.as_deref());

        let (wrapped_command, tracking_disabled) = if raw_mode || no_enter {
            (command.to_string(), true)
        } else {
            let end_marker = get_end_marker(&self.shell_type, &command_id);
            let start_marker = get_start_marker(&command_id);
            let wrapped = format!("echo \"{start_marker}\"; {command}; echo \"{end_marker}\"");
            (wrapped, false)
        };

        let execution = CommandExecution {
            id: command_id.clone(),
            pane_id: pane_id.to_string(),
            socket: resolved_socket.clone(),
            command: command.to_string(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: if tracking_disabled {
                Some("Tracking disabled for raw_mode or no_enter commands".to_string())
            } else {
                None
            },
            started_at: Instant::now(),
            completed_at: None,
            raw_mode,
            tracking_disabled,
        };

        {
            let mut commands = self.active_commands.write().await;
            commands.insert(command_id.clone(), execution);
        }

        self.cleanup_completed().await;

        if let Some(delay) = delay_ms {
            for ch in wrapped_command.chars() {
                tmux::send_keys(pane_id, &ch.to_string(), true, resolved_socket.as_deref()).await?;
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            if !no_enter {
                tmux::send_keys(pane_id, "Enter", false, resolved_socket.as_deref()).await?;
            }
        } else {
            // Send command as a whole (not literal/per-character)
            tmux::send_keys(pane_id, &wrapped_command, false, resolved_socket.as_deref()).await?;
            // Send Enter if needed
            if !no_enter {
                tmux::send_keys(pane_id, "Enter", false, resolved_socket.as_deref()).await?;
            }
        }

        Ok(command_id)
    }

    /// Check the status of a command by its ID.
    ///
    /// Returns `None` if the command ID is not found.
    /// Updates the command status based on captured pane output.
    pub async fn check_status(
        &self,
        command_id: &str,
        socket_override: Option<&str>,
    ) -> Result<Option<CommandExecution>> {
        self.cleanup_completed().await;

        let execution = {
            let commands = self.active_commands.read().await;
            commands.get(command_id).cloned()
        };

        let mut execution = match execution {
            Some(e) => e,
            None => return Ok(None),
        };

        match execution.status {
            CommandStatus::Completed | CommandStatus::Error => {
                return Ok(Some(execution));
            }
            CommandStatus::Pending if execution.raw_mode || execution.tracking_disabled => {
                return Ok(Some(execution));
            }
            _ => {}
        }

        #[cfg(test)]
        let _env_guard = EnvVarGuard::set("TMUX_MCP_TEST_COMMAND_ID", &execution.id);

        let mut capture_lines = self.tracking.capture_initial_lines.max(1);
        let max_lines = self.tracking.capture_max_lines.max(capture_lines);
        let backoff = self.tracking.capture_backoff_factor.max(1);

        loop {
            let captured_output = tmux::capture_pane(
                &execution.pane_id,
                Some(capture_lines),
                false,
                None,
                None,
                true,
                execution.socket.as_deref().or(socket_override),
            )
            .await?;

            if let Some((output, exit_code)) = parse_command_output(&captured_output, &execution.id)
            {
                execution.exit_code = Some(exit_code);
                execution.output = Some(output);
                execution.completed_at = Some(Instant::now());
                execution.status = if exit_code == 0 {
                    CommandStatus::Completed
                } else {
                    CommandStatus::Error
                };

                let mut commands = self.active_commands.write().await;
                commands.insert(command_id.to_string(), execution.clone());
                break;
            }

            // No DONE marker yet. If the START marker is visible, the command is
            // still running: DONE is echoed after START, so it lands within the
            // same (or a smaller) window once printed. Widening the capture
            // cannot surface a marker that does not exist yet, so leave the
            // command Pending and let the caller poll again. Caching a terminal
            // Error here would make it sticky via the early-return above and the
            // command could never reach Completed.
            if captured_output.contains(&get_start_marker(&execution.id)) {
                break;
            }

            // START is absent: it may simply have scrolled out of the captured
            // window. Widen and retry first. If the window cannot grow any
            // further (already at max_lines, or a degenerate backoff factor of 1
            // that never advances) fall through to the expiry decision instead
            // of re-capturing the same window forever.
            let widened = (capture_lines.saturating_mul(backoff)).min(max_lines);
            if widened > capture_lines {
                capture_lines = widened;
                continue;
            }

            // Even the widest window lacks the START marker. This is either a
            // high-output command still running (its START scrolled past
            // `capture_max_lines` and DONE is not printed yet) or genuinely lost
            // tracking. Capture-line exhaustion is not a timeout, so fall back to
            // a real time bound: stay Pending until the command has outlived the
            // tracking deadline, only then declare it expired. This keeps a slow,
            // verbose command recoverable (relax parse completes it once DONE
            // appears) while still bounding genuinely-lost commands.
            let deadline = Duration::from_secs(self.tracking.tracking_deadline_seconds);
            if execution.started_at.elapsed() <= deadline {
                break;
            }

            execution.status = CommandStatus::Error;
            execution.output =
                Some("tracking expired; markers not found in pane history".to_string());
            execution.completed_at = Some(Instant::now());

            let mut commands = self.active_commands.write().await;
            commands.insert(command_id.to_string(), execution.clone());
            break;
        }

        self.cleanup_completed().await;

        Ok(Some(execution))
    }

    /// Get a command by ID without updating its status.
    pub async fn get_command(&self, id: &str) -> Option<CommandExecution> {
        let commands = self.active_commands.read().await;
        commands.get(id).cloned()
    }

    /// Get all active command IDs.
    pub async fn get_active_ids(&self) -> Vec<String> {
        let commands = self.active_commands.read().await;
        commands.keys().cloned().collect()
    }

    /// Remove completed commands outside the configured retention window and count.
    async fn cleanup_completed(&self) {
        let retention_minutes = self.tracking.completed_retention_minutes;
        let retention_window = Duration::from_secs(retention_minutes.saturating_mul(60));
        let now = Instant::now();

        // A Pending command only transitions when it is polled (check_status).
        // A client that fires commands without ever polling would otherwise leak
        // entries forever, so drop Pending ones abandoned well past the tracking
        // deadline. The threshold (deadline + retention) is generous enough that a
        // client polling anywhere near the deadline still sees the proper
        // "expired" Error before the entry is reclaimed.
        let pending_abandon_window =
            Duration::from_secs(self.tracking.tracking_deadline_seconds) + retention_window;

        let mut commands = self.active_commands.write().await;
        commands.retain(|_, exec| {
            if exec.status == CommandStatus::Pending {
                let age = now
                    .checked_duration_since(exec.started_at)
                    .unwrap_or(Duration::ZERO);
                return age < pending_abandon_window;
            }
            let completed_at = match exec.completed_at {
                Some(instant) => instant,
                None => return true,
            };
            let age = now
                .checked_duration_since(completed_at)
                .unwrap_or(Duration::ZERO);
            age < retention_window
        });

        let max_entries = self.tracking.completed_max_entries as usize;
        let mut completed: Vec<(String, Instant)> = commands
            .iter()
            .filter_map(|(id, exec)| {
                if exec.status == CommandStatus::Pending {
                    return None;
                }
                exec.completed_at
                    .map(|completed_at| (id.clone(), completed_at))
            })
            .collect();

        if completed.len() <= max_entries {
            return;
        }

        completed.sort_by_key(|(_, completed_at)| *completed_at);
        let excess = completed.len().saturating_sub(max_entries);
        for (id, _) in completed.into_iter().take(excess) {
            commands.remove(&id);
        }
    }
}

/// Get the end marker command for the given shell type.
///
/// Fish shell uses `$status` for exit codes, while bash/zsh use `$?`.
pub fn get_start_marker(command_id: &str) -> String {
    format!("{START_MARKER_PREFIX}{command_id}")
}

fn end_marker_prefix(command_id: &str) -> String {
    format!("{END_MARKER_PREFIX}{command_id}_")
}

pub fn get_end_marker(shell: &ShellType, command_id: &str) -> String {
    let prefix = end_marker_prefix(command_id);
    match shell {
        ShellType::Fish => format!("{prefix}$status"),
        ShellType::Bash | ShellType::Zsh | ShellType::Unknown => format!("{prefix}$?"),
    }
}

/// Parse captured output to extract command output and exit code.
///
/// The DONE marker carries the exit code and is authoritative for completion.
/// The START marker only delimits where the command's output begins, so it is
/// optional: if it scrolled out of the captured window under heavy output, we
/// still complete the command from the DONE marker (the leading output is then
/// best-effort, bounded by whatever remains in the window).
/// Returns `None` if no DONE marker with a numeric exit code is present.
fn parse_command_output(captured: &str, command_id: &str) -> Option<(String, i32)> {
    let start_marker = get_start_marker(command_id);
    let after_start = match captured.rfind(&start_marker) {
        Some(start_idx) => &captured[start_idx + start_marker.len()..],
        None => captured,
    };

    // Find the LAST match of the end marker (not the first)
    // This is important because the pane output may contain the typed command line
    // (e.g., `echo TMUX_MCP_DONE_<id>_$?`) before the actual echoed output
    let end_prefix = end_marker_prefix(command_id);
    let end_regex = Regex::new(&format!(r"{}(\d+)", regex::escape(&end_prefix))).ok()?;
    let last_match = end_regex.captures_iter(after_start).last()?;

    let exit_code: i32 = last_match.get(1)?.as_str().parse().ok()?;

    let end_match = last_match.get(0)?;
    let output_end = end_match.start();

    let output = after_start[..output_end].trim().to_string();

    Some((output, exit_code))
}

/// Extract exit code from an end marker line.
#[allow(dead_code)]
fn extract_exit_code(line: &str, command_id: &str) -> Option<i32> {
    let end_prefix = end_marker_prefix(command_id);
    if line.contains(&end_prefix) {
        let end_regex = Regex::new(&format!(r"{}(\d+)", regex::escape(&end_prefix))).ok()?;
        let caps = end_regex.captures(line)?;
        caps.get(1)?.as_str().parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Error;
    use crate::test_support::TmuxStub;
    use crate::types::{CommandExecution, CommandStatus};
    use rstest::rstest;
    use std::time::Duration;
    use tempfile::tempdir;

    #[rstest]
    #[case(ShellType::Bash, "TMUX_MCP_DONE_cmd-1_$?")]
    #[case(ShellType::Zsh, "TMUX_MCP_DONE_cmd-1_$?")]
    #[case(ShellType::Fish, "TMUX_MCP_DONE_cmd-1_$status")]
    #[case(ShellType::Unknown, "TMUX_MCP_DONE_cmd-1_$?")]
    fn test_get_end_marker(#[case] shell: ShellType, #[case] expected: &str) {
        assert_eq!(get_end_marker(&shell, "cmd-1"), expected);
    }

    #[rstest]
    #[case("TMUX_MCP_DONE_cmd-1_0", Some(0))]
    #[case("TMUX_MCP_DONE_cmd-1_1", Some(1))]
    #[case("TMUX_MCP_DONE_cmd-1_127", Some(127))]
    #[case("TMUX_MCP_DONE_cmd-1_255", Some(255))]
    #[case("some output TMUX_MCP_DONE_cmd-1_42 more text", Some(42))]
    #[case("no marker here", None)]
    #[case("TMUX_MCP_DONE_cmd-1_", None)]
    #[case("TMUX_MCP_DONE_cmd-1_abc", None)]
    fn test_extract_exit_code(#[case] input: &str, #[case] expected: Option<i32>) {
        assert_eq!(extract_exit_code(input, "cmd-1"), expected);
    }

    #[rstest]
    #[case(
        "prompt$ TMUX_MCP_START_cmd-1\nhello world\nTMUX_MCP_DONE_cmd-1_0\nprompt$",
        Some(("hello world".to_string(), 0))
    )]
    #[case(
        "TMUX_MCP_START_cmd-1\nerror occurred\nTMUX_MCP_DONE_cmd-1_1",
        Some(("error occurred".to_string(), 1))
    )]
    #[case(
        "old TMUX_MCP_START_cmd-1\nold output\nTMUX_MCP_DONE_cmd-1_0\nnew TMUX_MCP_START_cmd-1\nnew output\nTMUX_MCP_DONE_cmd-1_2",
        Some(("new output".to_string(), 2))
    )]
    #[case(
        "TMUX_MCP_START_cmd-1\nline1\nline2\nline3\nTMUX_MCP_DONE_cmd-1_0",
        Some(("line1\nline2\nline3".to_string(), 0))
    )]
    #[case("no markers at all", None)]
    #[case("TMUX_MCP_START_cmd-1\nno end marker", None)]
    // START scrolled off but DONE is present: complete from DONE alone, with the
    // leading output best-effort (here, nothing precedes the marker).
    #[case(
        "TMUX_MCP_DONE_cmd-1_0\nno start marker",
        Some((String::new(), 0))
    )]
    #[case(
        "earlier output lost\nfinal line\nTMUX_MCP_DONE_cmd-1_3",
        Some(("earlier output lost\nfinal line".to_string(), 3))
    )]
    fn test_parse_command_output(#[case] input: &str, #[case] expected: Option<(String, i32)>) {
        assert_eq!(parse_command_output(input, "cmd-1"), expected);
    }

    #[rstest]
    #[case(
        "$ echo TMUX_MCP_START_cmd-1\nTMUX_MCP_START_cmd-1\n$ ls -la\ntotal 0\ndrwxr-xr-x  2 user user  40 Jan  1 00:00 .\ndrwxr-xr-x 10 user user 200 Jan  1 00:00 ..\n$ echo TMUX_MCP_DONE_cmd-1_$?\nTMUX_MCP_DONE_cmd-1_0\n$",
        Some(0)
    )]
    #[case(
        "TMUX_MCP_START_cmd-1\ncommand not found: foobar\nTMUX_MCP_DONE_cmd-1_127",
        Some(127)
    )]
    fn test_parse_realistic_output(#[case] input: &str, #[case] expected_exit: Option<i32>) {
        let result = parse_command_output(input, "cmd-1");
        match (result, expected_exit) {
            (Some((_, code)), Some(expected)) => assert_eq!(code, expected),
            (None, None) => {}
            (result, expected) => panic!("Expected {expected:?}, got {result:?}"),
        }
    }

    #[test]
    fn test_markers_are_correct() {
        assert_eq!(START_MARKER_PREFIX, "TMUX_MCP_START_");
        assert_eq!(END_MARKER_PREFIX, "TMUX_MCP_DONE_");
        assert_eq!(get_start_marker("cmd-1"), "TMUX_MCP_START_cmd-1");
    }

    #[rstest]
    fn test_command_tracker_new() {
        let tracker = CommandTracker::new(ShellType::Bash);
        assert!(matches!(tracker.shell_type, ShellType::Bash));
    }

    #[tokio::test]
    async fn execute_command_with_delay_sends_enter() {
        let _stub = TmuxStub::new();
        let tracker = CommandTracker::new(ShellType::Bash);

        let id = tracker
            .execute_command("%1", "echo hi", false, false, Some(0), None)
            .await
            .expect("execute command");

        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn execute_command_without_delay_sends_enter() {
        let _stub = TmuxStub::new();
        let tracker = CommandTracker::new(ShellType::Bash);

        let id = tracker
            .execute_command("%1", "echo hi", false, false, None, None)
            .await
            .expect("execute command");

        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn execute_command_returns_error_when_send_keys_fails() {
        let mut stub = TmuxStub::new();
        stub.set_var("TMUX_STUB_ERROR_CMD", "send-keys");
        let tracker = CommandTracker::new(ShellType::Bash);

        let err = tracker
            .execute_command("%1", "echo hi", false, false, None, None)
            .await
            .unwrap_err();

        match err {
            Error::Tmux { message } => assert!(message.contains("stub error")),
            _ => panic!("expected tmux error"),
        }
    }

    #[tokio::test]
    async fn check_status_returns_early_for_completed() {
        let tracker = CommandTracker::new(ShellType::Bash);
        let id = "completed-cmd".to_string();
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "echo done".into(),
            status: CommandStatus::Completed,
            exit_code: Some(0),
            output: Some("done".into()),
            started_at: Instant::now(),
            completed_at: Some(Instant::now()),
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let result = tracker.check_status(&id, None).await.expect("check status");
        assert!(matches!(
            result.map(|cmd| cmd.status),
            Some(CommandStatus::Completed)
        ));
    }

    #[tokio::test]
    async fn check_status_returns_none_for_unknown_id() {
        let tracker = CommandTracker::new(ShellType::Bash);
        let result = tracker
            .check_status("missing-command", None)
            .await
            .expect("check status");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn check_status_sets_error_on_nonzero_exit() {
        let mut stub = TmuxStub::new();
        let id = "error-cmd".to_string();
        stub.set_var(
            "TMUX_STUB_CAPTURE_OUTPUT",
            format!("TMUX_MCP_START_{id}\nbad\nTMUX_MCP_DONE_{id}_1\n"),
        );
        let tracker = CommandTracker::new(ShellType::Bash);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "false".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let result = tracker.check_status(&id, None).await.expect("check status");
        let status = result.map(|cmd| cmd.status).unwrap();
        assert_eq!(status, CommandStatus::Error);
    }

    #[tokio::test]
    async fn check_status_retries_capture_until_markers_found() {
        let mut stub = TmuxStub::new();
        let temp_dir = tempdir().expect("tempdir");
        let count_path = temp_dir.path().join("capture-count");
        let id = "retry-cmd".to_string();

        stub.set_var(
            "TMUX_STUB_CAPTURE_COUNT_FILE",
            count_path.to_str().expect("count path"),
        );
        stub.set_var("TMUX_STUB_CAPTURE_AFTER", "2");
        stub.set_var("TMUX_STUB_CAPTURE_BEFORE", "prompt\nno markers yet\n");
        stub.set_var(
            "TMUX_STUB_CAPTURE_AFTER_OUTPUT",
            format!("TMUX_MCP_START_{id}\nretry ok\nTMUX_MCP_DONE_{id}_0\n"),
        );

        let tracking = TrackingConfig {
            capture_initial_lines: 2,
            capture_max_lines: 4,
            capture_backoff_factor: 2,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "echo retry".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let result = tracker.check_status(&id, None).await.expect("check status");
        let command = result.expect("command");
        assert_eq!(command.status, CommandStatus::Completed);
        assert_eq!(command.exit_code, Some(0));
        assert_eq!(command.output.as_deref(), Some("retry ok"));

        let count = std::fs::read_to_string(&count_path)
            .expect("read count")
            .trim()
            .parse::<u32>()
            .expect("parse count");
        assert!(count >= 2);
    }

    #[tokio::test]
    async fn check_status_sets_error_when_markers_never_found() {
        let mut stub = TmuxStub::new();
        let id = "expired-cmd".to_string();
        stub.set_var("TMUX_STUB_CAPTURE_OUTPUT", "no markers here");

        // deadline 0: any elapsed time past start counts as expired, so the
        // markers-never-found path resolves to a terminal Error immediately.
        let tracking = TrackingConfig {
            capture_initial_lines: 1,
            capture_max_lines: 2,
            capture_backoff_factor: 2,
            tracking_deadline_seconds: 0,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "echo missing".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let result = tracker.check_status(&id, None).await.expect("check status");
        let command = result.expect("command");
        assert_eq!(command.status, CommandStatus::Error);
        assert_eq!(
            command.output.as_deref(),
            Some("tracking expired; markers not found in pane history")
        );
    }

    #[tokio::test]
    async fn check_status_stays_pending_when_start_lost_within_deadline() {
        // A high-output command can push its START marker past capture_max_lines
        // while still running. With no marker reachable but the command well
        // within its tracking deadline, check_status must stay Pending (not cache
        // a terminal Error) so it can still complete once DONE finally appears.
        let mut stub = TmuxStub::new();
        let id = "lost-start-cmd".to_string();
        stub.set_var("TMUX_STUB_CAPTURE_OUTPUT", "buried output, no markers");

        let tracking = TrackingConfig {
            capture_initial_lines: 1,
            capture_max_lines: 2,
            capture_backoff_factor: 2,
            tracking_deadline_seconds: 600,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "seq 1 100000".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let command = tracker
            .check_status(&id, None)
            .await
            .expect("check status")
            .expect("command");
        assert_eq!(command.status, CommandStatus::Pending);
        let stored = tracker.get_command(&id).await.expect("stored command");
        assert_eq!(stored.status, CommandStatus::Pending);
    }

    #[tokio::test]
    async fn check_status_completes_when_start_marker_scrolled_off() {
        // The DONE marker is authoritative: even if START scrolled out of the
        // captured window, a visible DONE with a numeric exit code completes the
        // command (leading output is best-effort).
        let mut stub = TmuxStub::new();
        let id = "done-only-cmd".to_string();
        stub.set_var(
            "TMUX_STUB_CAPTURE_OUTPUT",
            format!("...truncated output...\nlast line\nTMUX_MCP_DONE_{id}_0\n"),
        );

        let tracker = CommandTracker::new(ShellType::Bash);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "seq 1 100000".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let command = tracker
            .check_status(&id, None)
            .await
            .expect("check status")
            .expect("command");
        assert_eq!(command.status, CommandStatus::Completed);
        assert_eq!(command.exit_code, Some(0));
    }

    #[tokio::test]
    async fn check_status_terminates_with_degenerate_backoff_factor() {
        // Regression: capture_backoff_factor == 1 cannot widen the capture
        // window, so the escalation step never advances. The loop must still
        // terminate (fall through to the expiry decision) rather than spin
        // forever re-capturing the same window. deadline 0 makes it resolve to a
        // terminal Error immediately once it falls through; without the fix this
        // test hangs.
        let mut stub = TmuxStub::new();
        let id = "degenerate-backoff-cmd".to_string();
        stub.set_var("TMUX_STUB_CAPTURE_OUTPUT", "no markers here");

        let tracking = TrackingConfig {
            capture_initial_lines: 1,
            capture_max_lines: 16,
            capture_backoff_factor: 1,
            tracking_deadline_seconds: 0,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "echo missing".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        let command = tracker
            .check_status(&id, None)
            .await
            .expect("check status")
            .expect("command");
        assert_eq!(command.status, CommandStatus::Error);
    }

    #[tokio::test]
    async fn check_status_stays_pending_while_command_still_running() {
        // Regression: a command whose START marker is visible but whose DONE
        // marker has not been printed yet (still running) must NOT be cached as
        // a terminal Error. Otherwise the early-return path makes the bogus
        // Error sticky and the command can never reach Completed on later polls.
        let mut stub = TmuxStub::new();
        let id = "running-cmd".to_string();
        stub.set_var(
            "TMUX_STUB_CAPTURE_OUTPUT",
            format!("prompt\nTMUX_MCP_START_{id}\nstill working...\n"),
        );

        let tracking = TrackingConfig {
            capture_initial_lines: 1,
            capture_max_lines: 2,
            capture_backoff_factor: 2,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let execution = CommandExecution {
            id: id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "sleep 1".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: Instant::now(),
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(id.clone(), execution);
        }

        // First poll: START seen, DONE absent -> still Pending, not Error.
        let first = tracker
            .check_status(&id, None)
            .await
            .expect("check status")
            .expect("command");
        assert_eq!(first.status, CommandStatus::Pending);
        assert!(first.output.is_none());

        // The map entry must remain Pending so a later poll can still complete.
        let stored = tracker.get_command(&id).await.expect("stored command");
        assert_eq!(stored.status, CommandStatus::Pending);
    }

    #[tokio::test]
    async fn cleanup_completed_removes_old_and_preserves_pending() {
        let tracking = TrackingConfig {
            completed_retention_minutes: 1,
            completed_max_entries: 1000,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let old_id = "old".to_string();
        let new_id = "new".to_string();
        let pending_id = "pending".to_string();

        let now = Instant::now();
        let old_exec = CommandExecution {
            id: old_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "old".into(),
            status: CommandStatus::Completed,
            exit_code: Some(0),
            output: Some("old".into()),
            started_at: now,
            completed_at: Some(now - Duration::from_secs(120)),
            raw_mode: false,
            tracking_disabled: false,
        };
        let new_exec = CommandExecution {
            id: new_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "new".into(),
            status: CommandStatus::Completed,
            exit_code: Some(0),
            output: Some("new".into()),
            started_at: now,
            completed_at: Some(now),
            raw_mode: false,
            tracking_disabled: false,
        };
        let pending_exec = CommandExecution {
            id: pending_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "pending".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: now,
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(old_id.clone(), old_exec);
            commands.insert(new_id.clone(), new_exec);
            commands.insert(pending_id.clone(), pending_exec);
        }

        tracker.cleanup_completed().await;

        let commands = tracker.active_commands.read().await;
        assert!(!commands.contains_key(&old_id));
        assert!(commands.contains_key(&new_id));
        assert!(commands.contains_key(&pending_id));
    }

    #[tokio::test]
    async fn cleanup_completed_trims_to_max_entries() {
        let tracking = TrackingConfig {
            completed_retention_minutes: 10,
            completed_max_entries: 2,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);
        let oldest_id = "oldest".to_string();
        let middle_id = "middle".to_string();
        let newest_id = "newest".to_string();
        let pending_id = "pending".to_string();

        let now = Instant::now();
        let oldest_exec = CommandExecution {
            id: oldest_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "oldest".into(),
            status: CommandStatus::Completed,
            exit_code: Some(0),
            output: Some("oldest".into()),
            started_at: now,
            completed_at: Some(now - Duration::from_secs(180)),
            raw_mode: false,
            tracking_disabled: false,
        };
        let middle_exec = CommandExecution {
            id: middle_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "middle".into(),
            status: CommandStatus::Completed,
            exit_code: Some(0),
            output: Some("middle".into()),
            started_at: now,
            completed_at: Some(now - Duration::from_secs(120)),
            raw_mode: false,
            tracking_disabled: false,
        };
        let newest_exec = CommandExecution {
            id: newest_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "newest".into(),
            status: CommandStatus::Completed,
            exit_code: Some(0),
            output: Some("newest".into()),
            started_at: now,
            completed_at: Some(now - Duration::from_secs(60)),
            raw_mode: false,
            tracking_disabled: false,
        };
        let pending_exec = CommandExecution {
            id: pending_id.clone(),
            pane_id: "%1".into(),
            socket: None,
            command: "pending".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: now,
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };

        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert(oldest_id.clone(), oldest_exec);
            commands.insert(middle_id.clone(), middle_exec);
            commands.insert(newest_id.clone(), newest_exec);
            commands.insert(pending_id.clone(), pending_exec);
        }

        tracker.cleanup_completed().await;

        let commands = tracker.active_commands.read().await;
        assert!(!commands.contains_key(&oldest_id));
        assert!(commands.contains_key(&middle_id));
        assert!(commands.contains_key(&newest_id));
        assert!(commands.contains_key(&pending_id));
    }

    #[tokio::test]
    async fn cleanup_drops_abandoned_pending_but_keeps_recent() {
        // A Pending command transitions only when polled. cleanup must reclaim
        // ones abandoned past (deadline + retention) so a fire-and-forget client
        // cannot leak entries forever, while keeping recently-started ones.
        let tracking = TrackingConfig {
            tracking_deadline_seconds: 1,
            completed_retention_minutes: 0,
            ..TrackingConfig::default()
        };
        let tracker = CommandTracker::with_tracking(ShellType::Bash, tracking);

        let mk = |id: &str, started: Instant| CommandExecution {
            id: id.into(),
            pane_id: "%1".into(),
            socket: None,
            command: "sleep 100".into(),
            status: CommandStatus::Pending,
            exit_code: None,
            output: None,
            started_at: started,
            completed_at: None,
            raw_mode: false,
            tracking_disabled: false,
        };
        let stale_started = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .expect("backdate started_at");
        {
            let mut commands = tracker.active_commands.write().await;
            commands.insert("stale".into(), mk("stale", stale_started));
            commands.insert("fresh".into(), mk("fresh", Instant::now()));
        }

        tracker.cleanup_completed().await;

        let ids = tracker.get_active_ids().await;
        assert!(
            !ids.contains(&"stale".to_string()),
            "abandoned pending should be reclaimed"
        );
        assert!(
            ids.contains(&"fresh".to_string()),
            "recently-started pending should be kept"
        );
    }

    #[tokio::test]
    async fn concurrent_execute_command_tracks_unique_ids() {
        // Invariant: concurrent execute_command calls each get a distinct id and
        // none are lost from active_commands (the RwLock serializes the inserts).
        let _stub = TmuxStub::new();
        let tracker = Arc::new(CommandTracker::new(ShellType::Bash));

        let mut handles = Vec::new();
        for i in 0..16 {
            let tracker = Arc::clone(&tracker);
            handles.push(tokio::spawn(async move {
                tracker
                    .execute_command(&format!("%{i}"), "echo hi", false, false, None, None)
                    .await
                    .expect("execute command")
            }));
        }

        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.await.expect("join task"));
        }

        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "command ids must be unique");
        let tracked = tracker.get_active_ids().await;
        for id in &ids {
            assert!(tracked.contains(id), "command {id} must be tracked");
        }
    }
}
