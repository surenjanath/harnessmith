//! Line logging with an optional per-run sink.
//!
//! The Python code prints progress lines; here every line also reaches whichever run (if any)
//! the current thread is working for, so the web UI can stream a run's log. Worker threads
//! inherit the context through [`crate::util::par_map`].

use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A run's captured log lines.
#[derive(Clone, Default)]
pub struct Sink(pub Arc<Mutex<Vec<String>>>);

impl Sink {
    pub fn push(&self, line: String) {
        if let Ok(mut v) = self.0.lock() {
            v.push(line);
        }
    }
    pub fn lines_from(&self, start: usize) -> Vec<String> {
        self.0.lock().map(|v| v.iter().skip(start).cloned().collect()).unwrap_or_default()
    }
    pub fn len(&self) -> usize {
        self.0.lock().map(|v| v.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

static QUIET: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
static NAMED: Lazy<Mutex<HashMap<String, Sink>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Suppress stdout echo (lines still reach sinks).
pub fn set_quiet(q: bool) {
    *QUIET.lock().unwrap() = q;
}

/// The sink of the current thread, if any.
pub fn current() -> Option<Sink> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Install `sink` for the current thread; returns the previous one.
pub fn set_current(sink: Option<Sink>) -> Option<Sink> {
    CURRENT.with(|c| std::mem::replace(&mut *c.borrow_mut(), sink))
}

/// A process-wide named sink (used by the UI to look a run's log up by id).
pub fn named(name: &str) -> Sink {
    NAMED.lock().unwrap().entry(name.to_string()).or_default().clone()
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

#[macro_export]
macro_rules! say {
    ($($arg:tt)*) => { $crate::log::line(format!($($arg)*)) };
}
