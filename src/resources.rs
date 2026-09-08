//! Shared limits for blocking preparation and file input.
use anyhow::{Context, Result, ensure};
use std::{
    io::Read,
    path::Path,
    sync::{Arc, OnceLock},
};
use tokio::sync::Semaphore;

pub(crate) const MAX_IMAGE_INPUTS: usize = 16;
pub(crate) const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
pub(crate) const MAX_IMAGE_TOTAL_BYTES: usize = 64 * 1024 * 1024;

pub(crate) async fn run_blocking<T, F>(work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    static CAPACITY: OnceLock<Arc<Semaphore>> = OnceLock::new();
    let capacity = CAPACITY.get_or_init(|| Arc::new(Semaphore::new(4))).clone();
    run_blocking_with_capacity(capacity, work).await
}

async fn run_blocking_with_capacity<T, F>(capacity: Arc<Semaphore>, work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let permit = capacity
        .acquire_owned()
        .await
        .context("blocking capacity closed")?;
    tokio::task::spawn_blocking(move || {
        // Cancellation cannot release capacity while blocking work still runs.
        let _permit = permit;
        work()
    })
    .await
    .context("blocking task failed")?
}

pub(crate) fn read_file_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = open_regular_file(path)?;
    ensure!(
        file.metadata()?.len() <= limit as u64,
        "{} exceeds {limit} bytes",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(bytes)
}

fn open_regular_file(path: &Path) -> Result<std::fs::File> {
    #[cfg(unix)]
    let file: std::fs::File = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NONBLOCK | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .with_context(|| format!("opening {}", path.display()))?
    .into();
    #[cfg(not(unix))]
    let file = {
        ensure!(
            std::fs::metadata(path)?.is_file(),
            "{} is not a regular file",
            path.display()
        );
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?
    };
    ensure!(
        file.metadata()?.is_file(),
        "{} is not a regular file",
        path.display()
    );
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_limit_accepts_boundary_and_rejects_excess() {
        let path = std::env::temp_dir().join(format!("bounded-file-{}", std::process::id()));
        std::fs::write(&path, b"1234").unwrap();
        assert_eq!(read_file_limited(&path, 4).unwrap(), b"1234");
        assert!(read_file_limited(&path, 3).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[tokio::test]
    async fn cancelled_blocking_awaiter_keeps_capacity_until_work_exits() {
        let capacity = Arc::new(Semaphore::new(1));
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let active = tokio::spawn(run_blocking_with_capacity(capacity.clone(), move || {
            started.send(()).unwrap();
            // Dropping the sender also releases this worker if an assertion fails.
            let _ = wait.recv();
            Ok(())
        }));
        tokio::time::timeout(std::time::Duration::from_secs(2), ready)
            .await
            .unwrap()
            .unwrap();
        active.abort();
        assert!(active.await.unwrap_err().is_cancelled());
        assert_eq!(capacity.available_permits(), 0);
        assert!(capacity.try_acquire().is_err());

        release.send(()).unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_blocking_with_capacity(capacity.clone(), || Ok(42)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result, 42);
        assert_eq!(capacity.available_permits(), 1);
    }
}

#[cfg(all(test, unix))]
mod special_file_tests {
    use super::*;
    #[test]
    fn fifo_is_rejected_without_waiting_for_a_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.png");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let error = read_file_limited(&path, 100).unwrap_err();
        assert!(error.to_string().contains("not a regular file"));
    }
    #[test]
    fn device_is_rejected() {
        assert!(
            read_file_limited(Path::new("/dev/zero"), 100)
                .unwrap_err()
                .to_string()
                .contains("not a regular file")
        );
    }
}
