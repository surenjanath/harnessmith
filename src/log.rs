//! Line logging and structured progress events, with a per-run context.
//!
//! The Python code prints progress lines; here every line, and every structured event the
//! method emits (`event`), also reaches whichever run (if any) the current thread works for, so
//! the web UI can stream a run's log and draw its pipeline live. The context also carries the
//! current recursion mode and pipeline node, and a cancellation flag. Worker threads inherit
//! the context through [`crate::util::par_map`]. A sink may persist itself to disk.

use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct SinkInner {
    lines: Vec<String>,
    events: Vec<Value>,
    dir: Option<PathBuf>,
}

/// A run's captured log lines and events.
#[derive(Clone, Default)]
pub struct Sink {
    inner: Arc<Mutex<SinkInner>>,
    cancel: Arc<AtomicBool>,
}

impl Sink {
    /// A sink that also appends to `<dir>/log.txt` and `<dir>/events.jsonl`.
    pub fn persistent(dir: PathBuf) -> Sink {
        let _ = std::fs::create_dir_all(&dir);
        let s = Sink::default();
        s.inner.lock().unwrap().dir = Some(dir);
        s
    }
    /// Rebuild a finished run's sink from disk.
    pub fn load(dir: PathBuf) -> Sink {
        let s = Sink::default();
        {
            let mut g = s.inner.lock().unwrap();
            g.lines = std::fs::read_to_string(dir.join("log.txt")).unwrap_or_default().lines().map(|l| l.to_string()).collect();
            g.events = std::fs::read_to_string(dir.join("events.jsonl"))
                .unwrap_or_default()
                .lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
        }
        s
    }
    fn append(dir: &Option<PathBuf>, file: &str, line: &str) {
        if let Some(d) = dir {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(d.join(file)) {
                let _ = writeln!(f, "{line}");
            }
        }
    }
    pub fn push(&self, line: String) {
        if let Ok(mut g) = self.inner.lock() {
            Sink::append(&g.dir, "log.txt", &line);
            g.lines.push(line);
        }
    }
    pub fn push_event(&self, ev: Value) {
        if let Ok(mut g) = self.inner.lock() {
            Sink::append(&g.dir, "events.jsonl", &ev.to_string());
            g.events.push(ev);
        }
    }
    pub fn lines_from(&self, start: usize) -> Vec<String> {
        self.inner.lock().map(|g| g.lines.iter().skip(start).cloned().collect()).unwrap_or_default()
    }
    pub fn events_from(&self, start: usize) -> Vec<Value> {
        self.inner.lock().map(|g| g.events.iter().skip(start).cloned().collect()).unwrap_or_default()
    }
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.lines.len()).unwrap_or(0)
    }
    pub fn n_events(&self) -> usize {
        self.inner.lock().map(|g| g.events.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// What a thread is working on: the run's sink, and where in the pipeline it is.
#[derive(Clone, Default)]
pub struct Ctx {
    pub sink: Option<Sink>,
    pub mode: String,
    pub node: String,
    /// "edit" (an improver session), "run" (a solver rollout), "select", "learn".
    pub phase: String,
    /// The run's spending budget, shared by every agent in the run.
    pub budget: Option<std::sync::Arc<Budget>>,
}

/// A run's spending cap: every model call adds its cost; past the cap, agents stop.
#[derive(Debug, Default)]
pub struct Budget {
    pub cap_usd: f64,
    pub spent: Mutex<f64>,
    pub hit: std::sync::atomic::AtomicBool,
}

impl Budget {
    /// Add a cost; true when this crossed the cap.
    pub fn add(&self, usd: f64) -> bool {
        let mut s = self.spent.lock().unwrap();
        *s += usd;
        if self.cap_usd > 0.0 && *s >= self.cap_usd && !self.hit.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return true;
        }
        false
    }
    pub fn exceeded(&self) -> bool {
        self.hit.load(std::sync::atomic::Ordering::SeqCst)
    }
}

pub fn budget() -> Option<std::sync::Arc<Budget>> {
    CURRENT.with(|c| c.borrow().budget.clone())
}

pub fn set_budget(b: Option<std::sync::Arc<Budget>>) {
    CURRENT.with(|c| c.borrow_mut().budget = b);
}

thread_local! {
    static CURRENT: RefCell<Ctx> = RefCell::new(Ctx::default());
}

static QUIET: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
static NAMED: Lazy<Mutex<HashMap<String, Sink>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Suppress stdout echo (lines still reach sinks).
pub fn set_quiet(q: bool) {
    *QUIET.lock().unwrap() = q;
}

/// The full context of the current thread.
pub fn ctx() -> Ctx {
    CURRENT.with(|c| c.borrow().clone())
}

/// Install a full context; returns the previous one.
pub fn set_ctx(ctx: Ctx) -> Ctx {
    CURRENT.with(|c| std::mem::replace(&mut *c.borrow_mut(), ctx))
}

/// The sink of the current thread, if any.
pub fn current() -> Option<Sink> {
    CURRENT.with(|c| c.borrow().sink.clone())
}

/// Install `sink` for the current thread; returns the previous one.
pub fn set_current(sink: Option<Sink>) -> Option<Sink> {
    CURRENT.with(|c| std::mem::replace(&mut c.borrow_mut().sink, sink))
}

pub fn set_mode(mode: &str) {
    CURRENT.with(|c| c.borrow_mut().mode = mode.to_string());
}

pub fn set_node(node: &str) {
    CURRENT.with(|c| c.borrow_mut().node = node.to_string());
}

pub fn set_phase(phase: &str) {
    CURRENT.with(|c| c.borrow_mut().phase = phase.to_string());
}

/// Set node and phase together.
pub fn at(node: &str, phase: &str) {
    CURRENT.with(|c| {
        let mut c = c.borrow_mut();
        c.node = node.to_string();
        c.phase = phase.to_string();
    });
}

pub fn node() -> String {
    CURRENT.with(|c| c.borrow().node.clone())
}

/// Register a sink under a name so the UI can find it.
pub fn register(name: &str, sink: Sink) {
    NAMED.lock().unwrap().insert(name.to_string(), sink);
}

/// A process-wide named sink (created empty if unknown).
pub fn named(name: &str) -> Sink {
    NAMED.lock().unwrap().entry(name.to_string()).or_default().clone()
}

/// Whether the current run was cancelled.
pub fn cancelled() -> bool {
    current().map(|s| s.is_cancelled()).unwrap_or(false)
}

/// Emit one line.
pub fn line(s: impl Into<String>) {
    let s = s.into();
    if !*QUIET.lock().unwrap() {
        println!("{s}");
    }
    if let Some(sink) = current() {
        sink.push(s);
    }
}

/// Emit a structured progress event. `kind` is what happened (stage, rollout, select, learn,
/// step, task, ...), `status` its state; `extra` adds fields. Mode and node come from the
/// thread's context unless `extra` sets them.
pub fn event(kind: &str, status: &str, extra: Value) {
    let c = ctx();
    let Some(sink) = c.sink else { return };
    let mut ev = json!({
        "t": chrono::Utc::now().timestamp_millis(),
        "kind": kind,
        "status": status,
        "mode": c.mode,
        "node": c.node,
        "phase": c.phase,
    });
    if let Value::Object(m) = extra {
        for (k, v) in m {
            ev[k] = v;
        }
    }
    sink.push_event(ev);
}

#[macro_export]
macro_rules! say {
    ($($arg:tt)*) => { $crate::log::line(format!($($arg)*)) };
}
