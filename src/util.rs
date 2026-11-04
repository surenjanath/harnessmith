//! Small filesystem and concurrency helpers shared by every module.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use walkdir::WalkDir;

/// All files under `root`, sorted by their relative path (Python's `sorted(rglob("*"))`).
pub fn files_sorted(root: &Path) -> Vec<PathBuf> {
    if !root.is_dir() {
        return vec![];
    }
    let mut out: Vec<PathBuf> = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    out.sort();
    out
}

/// `p` relative to `root`, with forward slashes.
pub fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/")
}

/// Whether any path component is `__pycache__`.
pub fn in_pycache(p: &Path) -> bool {
    p.components().any(|c| c.as_os_str() == "__pycache__")
}

pub fn read_lossy(p: &Path) -> Result<String> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

pub fn write(p: &Path, body: impl AsRef<[u8]>) -> Result<()> {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(p, body).with_context(|| format!("writing {}", p.display()))
}

/// Copy a directory tree (`shutil.copytree(..., dirs_exist_ok=True)`).
pub fn copy_tree(src: &Path, dst: &Path) -> Result<usize> {
    copy_tree_filtered(src, dst, |_| true)
}

pub fn copy_tree_filtered(src: &Path, dst: &Path, keep: impl Fn(&str) -> bool) -> Result<usize> {
    std::fs::create_dir_all(dst)?;
    let mut n = 0;
    for p in files_sorted(src) {
        let r = rel(src, &p);
        if !keep(&r) {
            continue;
        }
        let t = dst.join(&r);
        if let Some(parent) = t.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&p, &t).with_context(|| format!("copying {}", p.display()))?;
        n += 1;
    }
    Ok(n)
}

pub fn rmtree(p: &Path) {
    if p.is_dir() {
        let _ = std::fs::remove_dir_all(p);
    } else if p.exists() {
        let _ = std::fs::remove_file(p);
    }
}

/// First `n` chars (not bytes) of `s`.
pub fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Last `n` chars of `s`.
pub fn tail_chars(s: &str, n: usize) -> String {
    let c = s.chars().count();
    s.chars().skip(c.saturating_sub(n)).collect()
}

pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Map `f` over `items` with at most `width` threads, preserving order. The caller's log sink
/// follows the work onto each worker thread.
pub fn par_map<T, R, F>(items: Vec<T>, width: usize, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(T) -> R + Sync,
{
    let width = width.max(1).min(items.len().max(1));
    if width <= 1 || items.len() <= 1 {
        return items.into_iter().map(&f).collect();
    }
    let sink = crate::log::current();
    let n = items.len();
    let queue = std::sync::Mutex::new(items.into_iter().enumerate().collect::<Vec<_>>().into_iter());
    let results: std::sync::Mutex<Vec<Option<R>>> =
        std::sync::Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..width {
            let sink = sink.clone();
            let (queue, results, f) = (&queue, &results, &f);
            s.spawn(move || {
                crate::log::set_current(sink);
                loop {
                    let next = queue.lock().unwrap().next();
                    let Some((i, item)) = next else { break };
                    let r = f(item);
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|r| r.expect("worker result")).collect()
}

/// Result of a subprocess run with a wall-clock limit.
pub struct Output {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// Run `cmd`, feeding `stdin`, killing it after `timeout`.
pub fn run_with_timeout(mut cmd: Command, stdin: Option<&str>, timeout: Duration) -> Result<Output> {
    use std::io::{Read, Write};
    use wait_timeout::ChildExt;
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().context("spawning subprocess")?;
    if let Some(input) = stdin {
        let mut si = child.stdin.take().unwrap();
        let input = input.to_string();
        std::thread::spawn(move || {
            let _ = si.write_all(input.as_bytes());
        });
    }
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = so.read_to_end(&mut b);
        b
    });
    let t_err = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = se.read_to_end(&mut b);
        b
    });
    let status = child.wait_timeout(timeout)?;
    let (code, timed_out) = match status {
        Some(s) => (s.code(), false),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            (None, true)
        }
    };
    let stdout = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).into_owned();
    Ok(Output { code, stdout, stderr, timed_out })
}

/// `which`: the first executable named `name` on PATH.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.is_file())
}

/// Python's `fnmatch.fnmatch` (POSIX: case-sensitive; `*` also matches `/`).
pub fn fnmatch(name: &str, pat: &str) -> bool {
    match glob::Pattern::new(pat) {
        Ok(p) => p.matches_with(
            name,
            glob::MatchOptions {
                case_sensitive: true,
                require_literal_separator: false,
                require_literal_leading_dot: false,
            },
        ),
        Err(_) => name == pat,
    }
}

/// Python-style `str.lstrip("./")`: strip any leading run of '.' and '/' chars.
pub fn lstrip_dot_slash(s: &str) -> &str {
    s.trim_start_matches(['.', '/'])
}

pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Atomically replace `path` with `body`.
pub fn write_atomic(path: &Path, body: &str) -> Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn slug_safe(label: &str) -> String {
    let re = regex::Regex::new(r"[^A-Za-z0-9_.-]+").unwrap();
    re.replace_all(label, "_").into_owned()
}
