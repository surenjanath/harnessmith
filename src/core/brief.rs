//! A one-page digest of the rollouts, computed in code, written to `context/brief.md`.
//!
//! An improver otherwise reads 100k+ tokens of trajectories to find out what went wrong; small
//! models give up and keep the base. The brief puts the facts first: what each rollout produced,
//! which calls failed or repeated, which self-heals fired, and what the deliverable review
//! still flags. Facts only -- it never suggests an edit, and it carries no grader verdicts.

use super::spec::RunView;
use crate::util;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// One rollout's digest.
pub fn rollout(task_prompt: &str, r: &RunView) -> String {
    let mut s = format!("## {}\n\n", r.label);
    let traj = util::read_lossy(&r.run_dir.join("trajectory.jsonl")).unwrap_or_default();
    let (mut calls, mut errors, mut heals, mut by_tool) = (0usize, vec![], vec![], BTreeMap::<String, usize>::new());
    let mut repeats: BTreeMap<String, usize> = BTreeMap::new();
    let mut finish = String::new();
    let mut session_error = String::new();
    for line in traj.lines() {
        let Ok(e) = serde_json::from_str::<Value>(line) else { continue };
        match e["type"].as_str() {
            Some("tool") => {
                calls += 1;
                let name = e["name"].as_str().unwrap_or("?").to_string();
                *by_tool.entry(name.clone()).or_default() += 1;
                *repeats.entry(format!("{name} {}", util::take_chars(&e["input"].to_string(), 80))).or_default() += 1;
                if e["status"] == "error" {
                    let first = e["output"].as_str().unwrap_or("").lines().find(|l| !l.trim().is_empty()).unwrap_or("").to_string();
                    errors.push(format!("`{name}` → {}", util::take_chars(&first, 160)));
                }
                if name == "finish" {
                    finish = e["input"]["answer"].as_str().unwrap_or("").to_string();
                }
            }
            Some("heal") => heals.push(e["kind"].as_str().unwrap_or("?").to_string()),
            Some("error") => session_error = e["message"].as_str().unwrap_or("").to_string(),
            _ => {}
        }
    }
    let out = r.run_dir.join("workspace/output");
    let files: Vec<String> = util::files_sorted(&out).iter().map(|p| format!("{} ({} bytes)", util::rel(&out, p), std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))).collect();
    s += &format!("- Outcome: {}{}\n", if r.delivered { "delivered" } else { "NOTHING delivered" }, if session_error.is_empty() { String::new() } else { format!("; session ended with: {session_error}") });
    s += &format!("- Files in output/: {}\n", if files.is_empty() { "(none)".into() } else { files.join(", ") });
    s += &format!("- Tool calls: {calls} ({})\n", by_tool.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join(", "));
    let reps: Vec<String> = repeats.iter().filter(|(_, n)| **n >= 3).map(|(k, n)| format!("`{k}` ×{n}")).collect();
    if !reps.is_empty() {
        s += &format!("- Repeated the same call: {}\n", reps.join("; "));
    }
    if !errors.is_empty() {
        s += &format!("- Failed calls ({}):\n", errors.len());
        for e in errors.iter().take(8) {
            s += &format!("  - {e}\n");
        }
        if errors.len() > 8 {
            s += &format!("  - …and {} more\n", errors.len() - 8);
        }
    }
    if !heals.is_empty() {
        s += &format!("- Self-heals: {}\n", heals.join(", "));
    }
    if out.is_dir() {
        let issues = crate::llm::review::review(task_prompt, &out);
        if !issues.is_empty() {
            s += "- Deliverable check (mechanical, against the task text):\n";
            for i in issues.iter().take(8) {
                s += &format!("  - {i}\n");
            }
        }
        let placeholders = placeholder_lines(&out);
        if !placeholders.is_empty() {
            s += &format!("- Placeholders left in the output: {}\n", placeholders.join("; "));
        }
    }
    if !finish.trim().is_empty() {
        s += &format!("- Final message: {}\n", util::take_chars(&finish.replace('\n', " "), 200));
    }
    s
}

/// Lines that still carry template placeholders such as `[Your Name]` or `[citation needed]`.
pub fn placeholder_lines(out: &Path) -> Vec<String> {
    let re = regex::Regex::new(r"(?i)\[(your [a-z ]+|date|name|citation needed|todo|tbd|insert [a-z ]+|placeholder)\]|lorem ipsum").unwrap();
    let mut hits = vec![];
    for p in util::files_sorted(out) {
        let body = std::fs::read_to_string(&p).unwrap_or_default();
        for m in re.find_iter(&body).take(3) {
            hits.push(format!("{} in {}", m.as_str(), util::rel(out, &p)));
        }
    }
    hits.sort();
    hits.dedup();
    hits.truncate(6);
    hits
}

/// The whole brief for a set of rollouts.
pub fn build(task_prompt: &str, rollouts: &[RunView]) -> String {
    let mut s = String::from("# BRIEF -- read this first\n\nA mechanical digest of the rollouts below, computed from their trajectories and output folders. It states facts only. Verify anything you act on in `context/rollouts/`, and cite the trajectory moment there.\n\nThe edits that help most target what went wrong here: a failed or repeated call, a missing file type, a placeholder, a step the solver kept rediscovering. An edit that does not address one of these is unlikely to change the next rollout.\n\n");
    for r in rollouts {
        s += &rollout(task_prompt, r);
        s += "\n";
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_lists_failures_repeats_and_placeholders() {
        let d = tempfile::tempdir().unwrap();
        let rd = d.path().join("base1");
        std::fs::create_dir_all(rd.join("workspace/output")).unwrap();
        std::fs::write(rd.join("workspace/output/report.md"), "# Report\nAuthor: [Your Name]\n").unwrap();
        let mut t = String::new();
        for _ in 0..3 {
            t += r#"{"type":"tool","name":"bash","input":{"command":"pandoc --version"},"status":"error","output":"exit_code=127\nbash: pandoc: command not found"}"#;
            t += "\n";
        }
        t += r#"{"type":"heal","kind":"loop_break"}"#;
        std::fs::write(rd.join("trajectory.jsonl"), t).unwrap();
        let r = RunView::new("base1", "x/base1", rd, true, true, "");
        let b = build("Write a PDF report", &[r]);
        assert!(b.contains("Failed calls (3)"), "{b}");
        assert!(b.contains("×3"), "{b}");
        assert!(b.contains("loop_break"), "{b}");
        assert!(b.contains("[Your Name]"), "{b}");
        assert!(b.contains(".pdf"), "{b}");
    }
}
