//! In-memory async job registry shared by the MCP tools.
//!
//! Generation jobs run on background tasks; this registry tracks their status
//! and result so `generate_image` can hand back a task id when a job outlives
//! the fast-return window, and `get_result` can fetch it later. It is
//! kind-agnostic (image today, video later) so the same registry is reused.
//!
//! Tasks are per server process and are lost on restart (stdio MCP servers are
//! per client session); any images already written stay on disk regardless.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::Mutex;

/// What a task produces. Video reuses this same async registry.
#[derive(Clone, Copy)]
pub enum TaskKind {
    Image,
    Video,
}

impl TaskKind {
    fn as_str(self) -> &'static str {
        match self {
            TaskKind::Image => "image",
            TaskKind::Video => "video",
        }
    }
}

enum Status {
    Pending,
    /// The lean result object (paths, dims, manifest, ...).
    Completed(Value),
    Failed(String),
}

struct TaskEntry {
    kind: TaskKind,
    status: Status,
    created_at: Instant,
    finished_at: Option<Instant>,
}

const MAX_RETAINED_TASKS: usize = 256;
const TERMINAL_TASK_TTL: Duration = Duration::from_secs(60 * 60);

/// A read-only view of a task for building a response.
pub struct TaskSnapshot {
    pub kind: &'static str,
    pub status: &'static str,
    pub result: Option<Value>,
    pub error: Option<String>,
}

/// Process-local registry of generation jobs, cheaply cloneable (shared `Arc`).
#[derive(Clone, Default)]
pub struct TaskRegistry {
    inner: Arc<Mutex<HashMap<String, TaskEntry>>>,
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new pending task.
    pub async fn insert_pending(&self, id: &str, kind: TaskKind) {
        let now = Instant::now();
        let mut entries = self.inner.lock().await;
        entries.insert(
            id.to_string(),
            TaskEntry {
                kind,
                status: Status::Pending,
                created_at: now,
                finished_at: None,
            },
        );
        prune_terminal(&mut entries, now);
    }

    /// Mark a task completed with its result.
    pub async fn complete(&self, id: &str, result: Value) {
        if let Some(entry) = self.inner.lock().await.get_mut(id) {
            entry.status = Status::Completed(result);
            entry.finished_at = Some(Instant::now());
        }
    }

    /// Mark a task failed with an error message.
    pub async fn fail(&self, id: &str, error: String) {
        if let Some(entry) = self.inner.lock().await.get_mut(id) {
            entry.status = Status::Failed(error);
            entry.finished_at = Some(Instant::now());
        }
    }

    /// Snapshot a task's current state, or `None` if the id is unknown.
    pub async fn snapshot(&self, id: &str) -> Option<TaskSnapshot> {
        let mut guard = self.inner.lock().await;
        prune_terminal(&mut guard, Instant::now());
        let entry = guard.get(id)?;
        Some(match &entry.status {
            Status::Pending => TaskSnapshot {
                kind: entry.kind.as_str(),
                status: "pending",
                result: None,
                error: None,
            },
            Status::Completed(v) => TaskSnapshot {
                kind: entry.kind.as_str(),
                status: "completed",
                result: Some(v.clone()),
                error: None,
            },
            Status::Failed(err) => TaskSnapshot {
                kind: entry.kind.as_str(),
                status: "failed",
                result: None,
                error: Some(err.clone()),
            },
        })
    }
}

/// Drop expired terminal tasks, then evict the oldest terminal results until
/// the registry is within its retention bound. Pending jobs are never evicted.
fn prune_terminal(entries: &mut HashMap<String, TaskEntry>, now: Instant) {
    entries.retain(|_, entry| {
        entry
            .finished_at
            .is_none_or(|finished| now.duration_since(finished) < TERMINAL_TASK_TTL)
    });

    while entries.len() > MAX_RETAINED_TASKS {
        let oldest = entries
            .iter()
            .filter(|(_, entry)| entry.finished_at.is_some())
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(id, _)| id.clone());
        match oldest {
            Some(id) => {
                entries.remove(&id);
            }
            None => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn lifecycle_pending_completed_failed_and_unknown() {
        let reg = TaskRegistry::new();
        assert!(reg.snapshot("missing").await.is_none());

        reg.insert_pending("a", TaskKind::Image).await;
        let s = reg.snapshot("a").await.unwrap();
        assert_eq!(s.status, "pending");
        assert_eq!(s.kind, "image");

        reg.complete("a", json!({"ok": true, "n": 1})).await;
        let s = reg.snapshot("a").await.unwrap();
        assert_eq!(s.status, "completed");
        assert_eq!(s.result.unwrap()["n"], 1);

        reg.insert_pending("b", TaskKind::Video).await;
        reg.fail("b", "boom".to_string()).await;
        let s = reg.snapshot("b").await.unwrap();
        assert_eq!(s.status, "failed");
        assert_eq!(s.kind, "video");
        assert_eq!(s.error.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn completed_tasks_are_evicted_at_the_retention_bound() {
        let reg = TaskRegistry::new();
        for i in 0..=MAX_RETAINED_TASKS {
            let id = format!("task-{i}");
            reg.insert_pending(&id, TaskKind::Image).await;
            reg.complete(&id, json!({"i": i})).await;
        }

        assert!(reg.snapshot("task-0").await.is_none());
        assert!(
            reg.snapshot(&format!("task-{MAX_RETAINED_TASKS}"))
                .await
                .is_some()
        );
    }
}
