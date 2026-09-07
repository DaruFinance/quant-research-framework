//! Run ownership using the same adjacent `.lock` sentinel as Python.
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;

thread_local! {
    static HELD: RefCell<HashSet<PathBuf>> = RefCell::new(HashSet::new());
}

pub(crate) struct LedgerGuard {
    path: PathBuf,
    file: Option<File>,
    // Drop must run on the thread that owns the reentrancy entry.
    _not_send: PhantomData<Rc<()>>,
}

impl LedgerGuard {
    pub(crate) fn acquire(path: &str) -> Self {
        let target = Path::new(path);
        let parent = target.parent().filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).expect("Cannot create trade ledger directory");
        let target = target.canonicalize().unwrap_or_else(|_| {
            parent.canonicalize().expect("Cannot resolve trade ledger directory")
                .join(target.file_name().expect("Trade ledger path needs a filename"))
        });
        let mut lock_name = target.into_os_string();
        lock_name.push(".lock");
        let path = PathBuf::from(lock_name);
        if HELD.with(|held| held.borrow().contains(&path)) {
            return Self { path, file: None, _not_send: PhantomData };
        }
        let file = OpenOptions::new().write(true).create_new(true).open(&path)
            .unwrap_or_else(|err| panic!(
                "Cannot lock trade ledger {}: {}. Choose a distinct BT_EXPORT_PATH \
                 (or Config.export_path). If its owner crashed, verify it has \
                 stopped before removing the lock.", path.display(), err));
        HELD.with(|held| { held.borrow_mut().insert(path.clone()); });
        let mut guard = Self { path, file: Some(file), _not_send: PhantomData };
        writeln!(guard.file.as_mut().unwrap(), "pid={}", std::process::id())
            .expect("Cannot write trade ledger lock owner");
        guard
    }
}

impl Drop for LedgerGuard {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            drop(file); // close first: Windows refuses to unlink open files
            if let Err(err) = fs::remove_file(&self.path) {
                eprintln!("Cannot remove trade ledger lock {}: {}", self.path.display(), err);
            }
            HELD.with(|held| { held.borrow_mut().remove(&self.path); });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn target() -> String {
        std::env::temp_dir().join(format!("qrf-ledger-{}-{}.csv",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)))
            .to_str().unwrap().to_owned()
    }

    #[test]
    fn nested_calls_keep_the_outer_lock_and_other_threads_fail() {
        let path = target();
        let outer = LedgerGuard::acquire(&path);
        let nested = LedgerGuard::acquire(&path);
        drop(nested);
        assert!(outer.path.exists());
        let other = path.clone();
        assert!(std::thread::spawn(move || {
            let _guard = LedgerGuard::acquire(&other);
        }).join().is_err());
        let lock = outer.path.clone();
        drop(outer);
        assert!(!lock.exists());
        let _again = LedgerGuard::acquire(&path);
    }

    #[test]
    fn panic_releases_ownership_and_distinct_paths_are_independent() {
        let path = target();
        assert!(std::panic::catch_unwind(|| {
            let _one = LedgerGuard::acquire(&path);
            let _two = LedgerGuard::acquire(&target());
            panic!("test failure");
        }).is_err());
        let _again = LedgerGuard::acquire(&path);
    }

    #[test]
    fn public_engine_entries_refuse_a_foreign_lock_before_touching_data() {
        use crate::{Config, RegimeConfig, classic_single_run, default_ema_signal,
                    run_cfg, run_with_regime_cfg, walk_forward_collect};
        let path = target();
        let lock = format!("{}.lock", path);
        fs::write(&lock, "pid=another\n").unwrap();
        for entry in 0..4 {
            let mut cfg = Config::new();
            cfg.export_path = path.clone();
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match entry {
                    0 => { classic_single_run(&[], &mut cfg, "test", default_ema_signal); }
                    1 => run_cfg(&[], "test", default_ema_signal, cfg),
                    2 => run_with_regime_cfg(&[], "test", default_ema_signal, RegimeConfig::default(), cfg),
                    _ => { walk_forward_collect(&[], &[], &mut cfg, "test", default_ema_signal); }
                }
            })).unwrap_err();
            let message = err.downcast_ref::<String>().unwrap();
            assert!(message.contains("Cannot lock trade ledger"), "{}", message);
            assert_eq!(fs::read_to_string(&lock).unwrap(), "pid=another\n");
        }
        fs::remove_file(lock).unwrap();
    }
}
