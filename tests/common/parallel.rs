//! Dep-free parallel map for the spawn-bound differential test harnesses.
//!
//! The heavy compat / selfdiff / diff-corpus suites are each a single
//! `#[test]` that loops over thousands of cases, spawning `jq-jit` and the
//! reference `jq` as subprocesses per case. That work is process-isolated
//! (each case touches only its own child processes and local buffers — no
//! shared in-process state), so it parallelises cleanly without reviving the
//! cranelift JIT race that historically forced `--test-threads=1` (resolved
//! by thread-localising the JIT pools, #173, and dropping the libjq FFI).
//!
//! `par_map` fans a closure across `available_parallelism()` scoped worker
//! threads using an atomic work-stealing cursor (load-balanced when per-case
//! cost varies), and returns results in the original input order so callers
//! keep their deterministic reporting. No `unsafe`, no external crates.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Apply `f` to every element of `items` across worker threads and return the
/// results in the original order. Falls back to a sequential map when there is
/// nothing to gain (single core, or 0/1 items).
pub fn par_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let len = items.len();
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(len.max(1));

    if workers <= 1 || len <= 1 {
        return items.iter().map(&f).collect();
    }

    let next = AtomicUsize::new(0);
    // Each worker drains the shared cursor and records (index, result) pairs
    // into its own local Vec, so no two threads write the same memory.
    let collected: Vec<Vec<(usize, R)>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let next = &next;
                let f = &f;
                scope.spawn(move || {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= len {
                            break;
                        }
                        local.push((i, f(&items[i])));
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Reassemble in input order.
    let mut out: Vec<Option<R>> = (0..len).map(|_| None).collect();
    for chunk in collected {
        for (i, r) in chunk {
            out[i] = Some(r);
        }
    }
    out.into_iter()
        .map(|slot| slot.expect("every index produced exactly one result"))
        .collect()
}
