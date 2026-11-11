//! Running model-written Python: the `python` tool and the answer engine both go through here.
//!
//! Small models write short programs far more reliably than they do arithmetic, but they write
//! them the way people do in a notebook: the last line is a bare expression, `input()` is called,
//! an `ANSWER:` label sits inside the code. The runner below absorbs those habits (it prints the
//! value of a trailing expression, runs with stdin closed, and reports errors as one short,
//! program-only traceback) so a working program is not lost to a formality.

use super::sandbox::Sandbox;
use crate::util;
use anyhow::Result;
use regex::Regex;
use std::path::Path;
use std::time::Duration;

/// Executes `program.py` like a notebook cell: a trailing expression is printed, and a failure
/// is reported as the failing line(s) of the program plus the exception, nothing else.
const RUNNER: &str = r#"import ast, sys, traceback
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
try:
    tree = ast.parse(src, filename="program.py")
except SyntaxError as e:
    print(f"SyntaxError: {e.msg} (line {e.lineno}): {(e.text or '').strip()}", file=sys.stderr)
    sys.exit(1)
g = {"__name__": "__main__", "__file__": "program.py"}
# The standard modules are there without an import, as in a notebook that has them loaded:
# a forgotten `import csv` should not cost an attempt.
for _m in ("csv", "json", "math", "re", "statistics", "datetime", "collections", "itertools", "os"):
    try:
        g[_m] = __import__(_m)
    except Exception:
        pass
try:
    from datetime import date, timedelta
    from collections import Counter, defaultdict
    g.update(date=date, timedelta=timedelta, Counter=Counter, defaultdict=defaultdict)
except Exception:
    pass
try:
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        exec(compile(tree, "program.py", "exec"), g)
        v = eval(compile(ast.Expression(last.value), "program.py", "eval"), g)
        if v is not None:
            print(v)
    else:
        exec(compile(tree, "program.py", "exec"), g)
except SystemExit:
    raise
except BaseException as e:
    lines = src.splitlines()
    for fr in traceback.extract_tb(e.__traceback__):
        if fr.filename == "program.py" and 0 < fr.lineno <= len(lines):
            print(f"  line {fr.lineno}: {lines[fr.lineno - 1].strip()}", file=sys.stderr)
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
"#;

/// Where the runner and the programs live inside a workspace (hidden from `list_files`).
pub const DIR: &str = ".hx";

#[derive(Debug, Clone, Default)]
pub struct PyOut {
    pub stdout: String,
    /// The short error report when the program failed; empty on success.
    pub error: String,
    pub timed_out: bool,
}

impl PyOut {
    pub fn ok(&self) -> bool {
        self.error.is_empty() && !self.timed_out
    }
    /// What the model is shown: output, then the error if any.
    pub fn render(&self) -> String {
        let mut s = self.stdout.trim_end().to_string();
        if !self.error.is_empty() {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(&self.error);
        }
        s
    }
}

pub fn available() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| util::which("python3").is_some())
}

/// The first fenced code block of a reply (an unclosed fence runs to the end), or the whole
/// reply when it is plainly bare code. `None` when there is no code in it.
pub fn extract_code(reply: &str) -> Option<String> {
    let fenced = Regex::new(r"(?s)```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)(?:```|\z)").unwrap();
    // Prefer a block tagged as Python; otherwise the first block that looks like code.
    let tagged = Regex::new(r"(?s)```[ \t]*(?:python3?|py)[ \t]*\r?\n(.*?)(?:```|\z)").unwrap();
    if let Some(c) = tagged.captures(reply) {
        return Some(c[1].to_string()).filter(|s| !s.trim().is_empty());
    }
    if let Some(c) = fenced.captures(reply) {
        let body = c[1].to_string();
        if looks_like_code(&body) {
            return Some(body);
        }
    }
    let t = reply.trim();
    if !t.contains("```") && looks_like_code(t) {
        return Some(t.to_string());
    }
    None
}

fn looks_like_code(s: &str) -> bool {
    let kw = Regex::new(r"(?m)^\s*(import |from \w+ import |print\(|def |for |while |with |if |[A-Za-z_]\w*\s*=[^=])").unwrap();
    let lines = s.lines().filter(|l| !l.trim().is_empty()).count().max(1);
    kw.find_iter(s).count() * 2 >= lines.min(6)
}

/// Remove what models add to a program that is not code: an `ANSWER: …` label, a stray
/// closing fence, a leading "python" word.
pub fn clean(code: &str) -> String {
    let label = Regex::new(r"^\s*(ANSWER|Answer|FINAL ANSWER|Final answer)\s*:").unwrap();
    let mut lines: Vec<&str> = code.lines().filter(|l| !label.is_match(l) && l.trim() != "```").collect();
    if lines.first().map(|l| matches!(l.trim(), "python" | "python3" | "py")).unwrap_or(false) {
        lines.remove(0);
    }
    lines.join("\n")
}

/// Run `code` with the workspace as the working directory (inside `sandbox` when given).
pub fn run(workspace: &Path, sandbox: Option<&Sandbox>, code: &str, timeout: Duration) -> Result<PyOut> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = workspace.join(DIR);
    std::fs::create_dir_all(&dir)?;
    let runner = dir.join("runner.py");
    if std::fs::read_to_string(&runner).map(|s| s != RUNNER).unwrap_or(true) {
        util::write(&runner, RUNNER)?;
    }
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let name = format!("{DIR}/program_{}_{n}.py", std::process::id());
    util::write(&workspace.join(&name), clean(code))?;
    let cmdline = format!("python3 {DIR}/runner.py {name}");
    let mut c = match sandbox {
        Some(sb) => sb.command(&cmdline),
        None => {
            let mut c = std::process::Command::new("bash");
            c.arg("-c").arg(&cmdline).current_dir(workspace);
            super::sandbox::scrub_env(&mut c);
            c
        }
    };
    c.env("PYTHONDONTWRITEBYTECODE", "1").env("PYTHONIOENCODING", "utf-8");
    let out = util::run_with_timeout(c, None, timeout)?;
    let _ = std::fs::remove_file(workspace.join(&name));
    let cap = |s: &str, n: usize| if util::char_len(s) > n { format!("{}\n…(output truncated)", util::take_chars(s, n)) } else { s.to_string() };
    let mut r = PyOut { stdout: cap(&out.stdout, 20_000), ..Default::default() };
    if out.timed_out {
        r.timed_out = true;
        r.error = format!("the program did not finish within {}s (an endless loop, or waiting for input)", timeout.as_secs());
    } else if out.code != Some(0) {
        r.error = explain(&cap(out.stderr.trim(), 2_000));
    }
    Ok(r)
}

/// Add the one hint that usually unblocks a small model after a given error.
fn explain(err: &str) -> String {
    let hint = if err.contains("EOFError") {
        "\nHint: input() cannot be used here. Put the values from the question directly in the program."
    } else if err.contains("ModuleNotFoundError") || err.contains("No module named") {
        "\nHint: that package is not installed. Use the standard library: csv, json, math, datetime, statistics, re, collections."
    } else if err.contains("FileNotFoundError") {
        "\nHint: only the files named in the task exist, in the current directory. Use the exact file name; if the data is in the question itself, put it in the program."
    } else if err.contains("SyntaxError") {
        "\nHint: write the logic as several short lines (a plain for-loop with an if inside, adding to a total), not as one long expression."
    } else if err.contains("NameError") {
        "\nHint: nothing is defined before your program runs. It must be complete: load each file it needs itself (for a CSV: rows = list(csv.DictReader(open(\"<the file name>\")))) and define every variable it uses."
    } else if err.contains("can't multiply sequence by non-int") || err.contains("unsupported operand type(s)") && err.contains("'str'") {
        "\nHint: values read from a CSV or a text file are strings. Convert before calculating: float(row[\"column\"]) or int(...)."
    } else if err.contains("KeyError") {
        "\nHint: print the available keys / column names first (for a CSV: print(rows[0].keys())), then use the exact spelling."
    } else if err.contains("could not convert string to float") || err.contains("invalid literal for int()") {
        "\nHint: that value is text, not a number. Do not convert a text column (a name, a region, an id): use it as it is, as a label or a dictionary key. Convert only the columns that hold numbers. csv.DictReader already skips the header row."
    } else {
        ""
    };
    format!("{err}{hint}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_and_cleans_code() {
        let r = "Here is the program:\n```python\nx = 2\nx * 21\n```\nANSWER: 42";
        assert_eq!(extract_code(r).unwrap().trim(), "x = 2\nx * 21");
        // an unclosed fence, as a reply cut off at the token limit
        assert_eq!(extract_code("```python\nprint(1)\n").unwrap().trim(), "print(1)");
        // bare code with no fence
        assert!(extract_code("import csv\nrows = list(csv.DictReader(open('a.csv')))\nprint(len(rows))").is_some());
        assert!(extract_code("The answer is 42 because the total equals the sum.").is_none());
        assert!(extract_code("ANSWER: 42").is_none());
        assert!(extract_code("total = 40 + 2\nprint(total)\nANSWER: 42").is_some());
        // an untagged block that is prose, not code
        assert!(extract_code("```\nThe total is 12\n```").is_none());
        assert_eq!(clean("total = 3\nANSWER: 3\n```"), "total = 3");
    }

    #[test]
    fn runs_like_a_notebook_and_reports_short_errors() {
        if !available() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("n.csv"), "a,b\n1,2\n3,4\n").unwrap();
        let t = Duration::from_secs(20);
        // trailing expression is printed; an ANSWER label inside the code is ignored
        let r = run(d.path(), None, "import csv\nrows = list(csv.DictReader(open('n.csv')))\ntotal = sum(int(r['a']) * int(r['b']) for r in rows)\ntotal\nANSWER: 99", t).unwrap();
        assert!(r.ok(), "{r:?}");
        assert_eq!(r.stdout.trim(), "14");
        // print() as the last statement is not printed twice
        assert_eq!(run(d.path(), None, "print(7)", t).unwrap().stdout.trim(), "7");
        // errors: the program's line and the exception, with a hint
        let r = run(d.path(), None, "x = int(input('n? '))\nprint(x)", t).unwrap();
        assert!(!r.ok() && r.error.contains("EOFError") && r.error.contains("line 1") && r.error.contains("Hint"), "{r:?}");
        let r = run(d.path(), None, "import csv\nrows = list(csv.DictReader(open('n.csv')))\nprint(rows[0]['zz'])", t).unwrap();
        assert!(r.error.contains("KeyError") && r.error.contains("line 3"), "{r:?}");
        let r = run(d.path(), None, "print(1", t).unwrap();
        assert!(r.error.contains("SyntaxError"), "{r:?}");
        // the standard modules are there without an import
        let r = run(d.path(), None, "rows = list(csv.DictReader(open('n.csv')))\nprint(sum(float(r['a']) for r in rows), math.floor(2.5), date(2024, 3, 1) - timedelta(days=1), Counter('aab')['a'])", t).unwrap();
        assert_eq!(r.stdout.trim(), "4.0 2 2024-02-29 2", "{r:?}");
        let r = run(d.path(), None, "print(sum(rows))", t).unwrap();
        assert!(r.error.contains("NameError") && r.error.contains("must be complete"), "{r:?}");
        let r = run(d.path(), None, "rows = list(csv.DictReader(open('n.csv')))\nprint(rows[0]['a'] * rows[0]['b'])", t).unwrap();
        assert!(r.error.contains("are strings"), "{r:?}");
        assert!(!d.path().join(DIR).read_dir().unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("program_")), "programs are cleaned up");
    }
}
