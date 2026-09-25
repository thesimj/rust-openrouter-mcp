//! In-memory async job registry shared by the MCP tools.
//!
//! Generation jobs run on background tasks; this registry tracks their status
//! and result so `generate_image` and `generate_video` can hand back a task id
//! when a job outlives the fast-return window, and `get_result` can fetch it
//! later. It is kind-agnostic (image and video) so the same registry is reused.
//!
//! Tasks are per server process and are lost on restart (stdio MCP servers are
//! per client session); any images or clips already written stay on disk
//! regardless.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::{sync::oneshot, task::JoinSet};

/// What a task produces. Video reuses this same async registry.
#[derive(Clone, Copy)]
pub enum TaskKind {
    Image,
    Video,
}

impl TaskKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            TaskKind::Image => "image",
            TaskKind::Video => "video",
        }
    }
}

/// Where a task is in its lifecycle.
#[derive(Clone, Debug)]
pub enum Status {
    Pending,
    /// The lean result object (paths, dims, manifest, ...).
    Completed(Value),
    Failed(String),
}

struct TaskEntry {
    kind: TaskKind,
    status: Status,
    finished_at: Option<Instant>,
}

const MAX_RETAINED_TASKS: usize = 256;
const MAX_PENDING_TASKS: usize = 32;
const TERMINAL_TASK_TTL: Duration = Duration::from_secs(60 * 60);

impl Status {
    /// The status name a response reports.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Status::Pending => "pending",
            Status::Completed(_) => "completed",
            Status::Failed(_) => "failed",
        }
    }
}

/// A read-only view of a task for building a response.
pub struct TaskSnapshot {
    pub kind: TaskKind,
    pub status: Status,
}

/// Process-local registry of generation jobs, cheaply cloneable (shared `Arc`).
#[derive(Clone, Default)]
pub struct TaskRegistry {
    inner: Arc<Mutex<HashMap<String, TaskEntry>>>,
    supervisor: Arc<Mutex<Supervisor>>,
}

#[derive(Default)]
struct Supervisor {
    jobs: JoinSet<()>,
    closed: bool,
}

/// Owns admission while preparing and running a job. Drop always releases it.
pub(crate) struct JobReservation {
    pub(crate) id: String,
    pub(crate) kind: TaskKind,
    entries: Arc<Mutex<HashMap<String, TaskEntry>>>,
    armed: bool,
}

impl JobReservation {
    fn finish(&mut self, result: Result<Value, String>) {
        if let Some(entry) = self.entries.lock().unwrap().get_mut(&self.id) {
            entry.status = match result {
                Ok(v) => Status::Completed(v),
                Err(e) => Status::Failed(e),
            };
            entry.finished_at = Some(Instant::now());
        }
        self.armed = false;
    }
}

impl Drop for JobReservation {
    fn drop(&mut self) {
        if self.armed {
            self.finish(Err("generation job cancelled or panicked".into()));
        }
    }
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reserve(&self, kind: TaskKind) -> Option<JobReservation> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let supervisor = self.supervisor.lock().unwrap();
        if supervisor.closed {
            return None;
        }
        let id = format!(
            "task-{}",
            NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        if !self.insert(&id, kind) {
            return None;
        }
        Some(JobReservation {
            id,
            kind,
            entries: self.inner.clone(),
            armed: true,
        })
    }

    pub(crate) fn start<F>(
        &self,
        mut reservation: JobReservation,
        run: F,
    ) -> Option<oneshot::Receiver<()>>
    where
        F: Future<Output = Result<Value, String>> + Send + 'static,
    {
        let mut supervisor = self.supervisor.lock().unwrap();
        if supervisor.closed {
            return None;
        }
        while supervisor.jobs.try_join_next().is_some() {}
        let (done, receiver) = oneshot::channel();
        supervisor.jobs.spawn(async move {
            reservation.finish(run.await);
            let _ = done.send(());
        });
        Some(receiver)
    }

    pub(crate) fn close_admission(&self) {
        self.supervisor.lock().unwrap().closed = true;
    }

    pub(crate) async fn shutdown(&self, grace: Duration) {
        let mut jobs = {
            let mut supervisor = self.supervisor.lock().unwrap();
            supervisor.closed = true;
            std::mem::take(&mut supervisor.jobs)
        };
        if tokio::time::timeout(grace, async { while jobs.join_next().await.is_some() {} })
            .await
            .is_err()
        {
            jobs.abort_all();
            while jobs.join_next().await.is_some() {}
        }
    }

    /// Register a new pending task.
    #[cfg(test)]
    pub async fn insert_pending(&self, id: &str, kind: TaskKind) -> bool {
        self.insert(id, kind)
    }

    fn insert(&self, id: &str, kind: TaskKind) -> bool {
        let now = Instant::now();
        let mut entries = self.inner.lock().unwrap();
        if entries
            .values()
            .filter(|entry| matches!(entry.status, Status::Pending))
            .count()
            >= MAX_PENDING_TASKS
        {
            return false;
        }
        entries.insert(
            id.to_string(),
            TaskEntry {
                kind,
                status: Status::Pending,
                finished_at: None,
            },
        );
        prune_terminal(&mut entries, now);
        true
    }

    /// Mark a task completed with its result.
    #[cfg(test)]
    pub async fn complete(&self, id: &str, result: Value) {
        if let Some(entry) = self.inner.lock().unwrap().get_mut(id) {
            entry.status = Status::Completed(result);
            entry.finished_at = Some(Instant::now());
        }
    }

    /// Mark a task failed with an error message.
    #[cfg(test)]
    pub async fn fail(&self, id: &str, error: String) {
        if let Some(entry) = self.inner.lock().unwrap().get_mut(id) {
            entry.status = Status::Failed(error);
            entry.finished_at = Some(Instant::now());
        }
    }

    /// Snapshot a task's current state, or `None` if the id is unknown.
    ///
    /// Reads before pruning, deliberately: when every other entry is pending,
    /// the job that just completed is the only terminal one and so the one the
    /// cap loop evicts. Pruning first meant a caller could evict the very
    /// result it was asking for - so a finished job answered "unknown task_id"
    /// while its output sat on disk. The prune still runs on every call, just
    /// one step later.
    pub async fn snapshot(&self, id: &str) -> Option<TaskSnapshot> {
        let mut guard = self.inner.lock().unwrap();
        let snap = guard.get(id).map(|entry| TaskSnapshot {
            kind: entry.kind,
            status: entry.status.clone(),
        });
        prune_terminal(&mut guard, Instant::now());
        snap
    }
}

impl TaskSnapshot {
    /// A "still running" snapshot for a task the registry can no longer show.
    pub(crate) fn pending(kind: TaskKind) -> Self {
        Self {
            kind,
            status: Status::Pending,
        }
    }
}

/// Drop expired terminal tasks, then evict the results that finished longest
/// ago until the registry is within its retention bound. Pending jobs are
/// never evicted.
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
            .min_by_key(|(_, entry)| entry.finished_at)
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
        assert!(matches!(s.status, Status::Pending));
        assert_eq!(s.kind.as_str(), "image");

        reg.complete("a", json!({"ok": true, "n": 1})).await;
        let s = reg.snapshot("a").await.unwrap();
        let Status::Completed(result) = &s.status else {
            panic!("not completed: {:?}", s.status);
        };
        assert_eq!(result["n"], 1);

        reg.insert_pending("b", TaskKind::Video).await;
        reg.fail("b", "boom".to_string()).await;
        let s = reg.snapshot("b").await.unwrap();
        assert!(matches!(&s.status, Status::Failed(e) if e == "boom"));
        assert_eq!(s.kind.as_str(), "video");
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

    /// Over the bound, the result that finished longest ago goes first - not
    /// the task created first. A long job is created early and finishes late;
    /// evicting it by creation time lost a result seconds after it was ready.
    #[test]
    fn prune_evicts_the_result_that_finished_longest_ago() {
        let start = Instant::now();
        let finished_at = |seconds: u64| TaskEntry {
            kind: TaskKind::Image,
            status: Status::Completed(json!({})),
            finished_at: Some(start + Duration::from_secs(seconds)),
        };
        // The long video was created first (task ids are sequential) but
        // finished last.
        let mut entries = HashMap::new();
        entries.insert("task-1".to_string(), finished_at(100));
        for i in 2..=MAX_RETAINED_TASKS + 1 {
            entries.insert(format!("task-{i}"), finished_at(11));
        }
        prune_terminal(&mut entries, start + Duration::from_secs(101));
        assert_eq!(entries.len(), MAX_RETAINED_TASKS);
        assert!(entries.contains_key("task-1"));
    }

    /// A poll must not evict the entry it is polling for. `snapshot` prunes on
    /// every call and the cap loop evicts a *terminal* entry - so
    /// pruning before the lookup meant asking about a just-finished job was what
    /// destroyed it, and the caller got `unknown task_id` while the generated
    /// file sat on disk. Reads first, prunes second.
    ///
    /// Getting over the bound takes pending jobs: `insert_pending` prunes too,
    /// but pending entries are never evicted, so a burst of in-flight jobs is
    /// what holds the registry above `MAX_RETAINED_TASKS`. That is precisely the
    /// state a long-running generation completes into.
    #[tokio::test]
    async fn polling_a_just_finished_task_does_not_evict_it() {
        let reg = TaskRegistry::new();
        // Construct an overfull legacy state directly; admission now prevents it.
        {
            let mut entries = reg.inner.lock().unwrap();
            for i in 0..=MAX_RETAINED_TASKS {
                entries.insert(
                    format!("task-{i}"),
                    TaskEntry {
                        kind: TaskKind::Video,
                        status: Status::Pending,
                        finished_at: None,
                    },
                );
            }
        }
        // Nothing is terminal yet, so the registry sits one over the bound.
        reg.complete("task-0", json!({"i": 0})).await;

        let snap = reg.snapshot("task-0").await;
        assert!(
            snap.is_some(),
            "polling task-0 evicted it before answering the question"
        );
        let snap = snap.unwrap();
        let Status::Completed(result) = &snap.status else {
            panic!("not completed: {:?}", snap.status);
        };
        assert_eq!(result["i"], 0);
        // The prune still runs, just after the read: now that task-0 is terminal
        // and the registry is over its bound, the next call reclaims it.
        assert!(reg.snapshot("task-0").await.is_none());
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
    #[tokio::test]
    async fn pending_limit_rejects_work_and_reopens_after_completion() {
        let registry = TaskRegistry::new();
        for i in 0..MAX_PENDING_TASKS {
            assert!(
                registry
                    .insert_pending(&i.to_string(), TaskKind::Image)
                    .await
            );
        }
        assert!(!registry.insert_pending("excess", TaskKind::Video).await);
        assert!(registry.snapshot("excess").await.is_none());
        registry.complete("0", serde_json::json!({})).await;
        assert!(registry.insert_pending("next", TaskKind::Video).await);
    }
    #[tokio::test]
    async fn dropping_reservation_marks_failure_and_reopens_capacity() {
        let registry = TaskRegistry::new();
        let mut reservations: Vec<_> = (0..MAX_PENDING_TASKS)
            .map(|_| registry.reserve(TaskKind::Image).unwrap())
            .collect();
        assert!(registry.reserve(TaskKind::Image).is_none());
        let released = reservations.pop().unwrap();
        let id = released.id.clone();
        drop(released);
        let snapshot = registry.snapshot(&id).await.unwrap();
        assert!(matches!(&snapshot.status, Status::Failed(e) if e.contains("cancelled")));
        assert!(registry.reserve(TaskKind::Video).is_some());
    }

    #[tokio::test]
    async fn panicked_job_becomes_failed_and_can_be_drained() {
        let registry = TaskRegistry::new();
        let reservation = registry.reserve(TaskKind::Video).unwrap();
        let id = reservation.id.clone();
        let done = registry
            .start(reservation, async { panic!("synthetic job panic") })
            .unwrap();
        assert!(done.await.is_err());
        registry.shutdown(Duration::from_secs(1)).await;
        let snapshot = registry.snapshot(&id).await.unwrap();
        assert!(matches!(&snapshot.status, Status::Failed(e) if e.contains("panicked")));
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_drains_completed_work_before_grace_expires() {
        let registry = TaskRegistry::new();
        let reservation = registry.reserve(TaskKind::Image).unwrap();
        let id = reservation.id.clone();
        let done = registry
            .start(reservation, async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(serde_json::json!({"saved": true}))
            })
            .unwrap();
        let started = tokio::time::Instant::now();
        registry.shutdown(Duration::from_secs(10)).await;
        assert_eq!(started.elapsed(), Duration::from_secs(2));
        done.await.unwrap();
        let snapshot = registry.snapshot(&id).await.unwrap();
        let Status::Completed(result) = &snapshot.status else {
            panic!("not completed: {:?}", snapshot.status);
        };
        assert_eq!(result["saved"], true);
        assert!(registry.reserve(TaskKind::Image).is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_aborts_at_deadline_and_records_failure() {
        let registry = TaskRegistry::new();
        let reservation = registry.reserve(TaskKind::Video).unwrap();
        let id = reservation.id.clone();
        let done = registry.start(reservation, std::future::pending()).unwrap();
        let started = tokio::time::Instant::now();
        registry.shutdown(Duration::from_secs(5)).await;
        assert_eq!(started.elapsed(), Duration::from_secs(5));
        assert!(done.await.is_err());
        let snapshot = registry.snapshot(&id).await.unwrap();
        assert!(matches!(&snapshot.status, Status::Failed(e) if e.contains("cancelled")));
        assert!(registry.reserve(TaskKind::Image).is_none());
    }

    #[tokio::test]
    async fn closing_admission_rejects_new_and_prepared_jobs() {
        let registry = TaskRegistry::new();
        let reservation = registry.reserve(TaskKind::Image).unwrap();
        let id = reservation.id.clone();
        registry.close_admission();
        assert!(registry.reserve(TaskKind::Video).is_none());
        assert!(
            registry
                .start(reservation, async {
                    panic!("closed registry must not run this job")
                })
                .is_none()
        );
        assert!(matches!(
            registry.snapshot(&id).await.unwrap().status,
            Status::Failed(_)
        ));
        registry.shutdown(Duration::from_secs(1)).await;
    }
}
