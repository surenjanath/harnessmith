//! The evidence tree an agentic proposer reads. The only way evidence reaches a model.
//!
//! ```text
//!     context/task.md                  the task as the agent received it
//!     context/rollouts/<label>.md      full trajectory -- no prompt-budget truncation
//!     context/deliverables/<label>.md  the artifact, via the judge's own extractor
//!     context/harness/<rel>            the c1..c8 tree being edited
//!     context/library/INDEX.md         earlier tasks in this domain + their full harnesses
//!     context/pool/<label>/            candidate harness + its rollout (aggregate / select only)
//!     MISSION.md                       what to do with all of it
//! ```
//!
//! Evidence is served as files and the prompt is a short pointer, so nothing is truncated to
//! fit a prompt. `mission()` builds that pointer.

use super::spec::{BenchmarkAdapter, RunView};
use super::tree::{self, Tree};
use crate::util;
use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A built evidence tree, plus what actually landed in it.
#[derive(Debug, Clone)]
pub struct ContextTree {
    pub root: PathBuf,
    pub task_id: String,
    pub rollouts: Vec<String>,
    pub deliverables: Vec<String>,
    pub harness_files: usize,
    pub library: bool,
    pub pool: Vec<String>,
    pub failed: Vec<String>,
    pub deliverables_mode: String,
    pub source_files: usize,
    pub decisions: usize,
    pub brief: bool,
}

impl ContextTree {
    /// The evidence-layout listing that goes in the mission; only mentions what exists.
    pub fn layout(&self) -> String {
        let mut lines = vec![];
        if self.brief {
            lines.push("context/brief.md                READ FIRST: a mechanical digest of every rollout (failures, repeats, missing files)".to_string());
        }
        lines.push("context/task.md                  the task as the agent received it".to_string());
        if self.source_files > 0 {
            lines.push("context/source/                 task source materials".into());
        }
        if self.decisions > 0 {
            lines.push("context/decisions/              earlier edits and composition decisions".into());
        }
        if !self.rollouts.is_empty() {
            let names = self.rollouts.iter().take(6).map(|r| format!("{r}.md")).collect::<Vec<_>>().join(", ");
            let more = if self.rollouts.len() > 6 {
                format!(" (+{} more)", self.rollouts.len() - 6)
            } else {
                String::new()
            };
            lines.push(format!("context/rollouts/               FULL trajectories: {names}{more}"));
        }
        if !self.deliverables.is_empty() {
            if self.deliverables_mode == "names" {
                lines.push("context/deliverables/<label>/    INDEX.md lists WHICH files each rollout produced -- names only.".into());
                lines.push("                                 File contents are deliberately not provided on this benchmark; work from the trajectories and the file set itself.".into());
            } else {
                lines.push("context/deliverables/<label>/    the files each rollout produced, one per artifact (browse INDEX.md for the set)".into());
            }
        }
        if self.harness_files > 0 {
            lines.push(format!(
                "context/harness/                the harness you are editing ({} files)",
                self.harness_files
            ));
        }
        if !self.pool.is_empty() {
            lines.push(format!(
                "context/pool/                   {} candidates, each with its harness and its rollout",
                self.pool.len()
            ));
        }
        if self.library {
            lines.push("context/library/                earlier tasks in this area -- browse INDEX.md first, then open what looks relevant".into());
        }
        lines.join("\n")
    }
}

/// One pool entry: (label, candidate harness dir, its rollout).
pub type PoolEntry = (String, PathBuf, Option<RunView>);

/// Write the evidence tree; return what landed, so the mission describes only that.
#[allow(clippy::too_many_arguments)]
pub fn build(
    dest: &Path,
    adapter: &dyn BenchmarkAdapter,
    task_id: &str,
    rollouts: &[RunView],
    harness_dir: Option<&Path>,
    library_dir: Option<&Path>,
    pool: &[PoolEntry],
    decisions: Option<&BTreeMap<String, String>>,
    obs_trunc: Option<usize>,
) -> Result<ContextTree> {
    util::rmtree(dest);
    std::fs::create_dir_all(dest)?;
    let trunc = obs_trunc.unwrap_or(adapter.spec().obs_trunc);
    util::write(&dest.join("task.md"), adapter.task_prompt(task_id)?)?;
    let source_files = copy_source(adapter, task_id, dest)?;
    for (label, notes) in decisions.into_iter().flatten() {
        let body = if notes.is_empty() { "No decision notes returned." } else { notes.as_str() };
        util::write(&dest.join("decisions").join(format!("{label}.md")), body)?;
    }
    let (mut got_r, mut got_d, mut failed_r) = (vec![], vec![], vec![]);
    if !rollouts.is_empty() {
        std::fs::create_dir_all(dest.join("rollouts"))?;
        std::fs::create_dir_all(dest.join("deliverables"))?;
    }
    let names_only = adapter.spec().proposer_deliverables == "names";
    for r in rollouts {
        let failure = write_rollout(dest, r, adapter, trunc, names_only)?;
        if failure.is_empty() {
            got_r.push(r.label.clone());
        } else {
            failed_r.push(failure);
        }
        got_d.push(r.label.clone());
    }
    let mut n_h = 0;
    if let Some(h) = harness_dir.filter(|h| h.is_dir()) {
        assert_no_leak_paths(h, &adapter.spec().leak_globs)?;
        n_h = copy_component_tree(h, &dest.join("harness"))?;
    }
    let got_p = write_pool(dest, pool, adapter, trunc, names_only)?;
    let mut brief = false;
    if adapter.spec().improver_brief {
        let mut all: Vec<RunView> = rollouts.to_vec();
        all.extend(pool.iter().filter_map(|(l, _, r)| r.clone().map(|mut r| {
            r.label = format!("{l} (candidate)");
            r
        })));
        if !all.is_empty() {
            util::write(&dest.join("brief.md"), super::brief::build(&adapter.task_prompt(task_id)?, &all))?;
            brief = true;
        }
    }
    let mut lib = false;
    if let Some(l) = library_dir {
        if l.is_dir() && std::fs::read_dir(l).map(|mut d| d.next().is_some()).unwrap_or(false) {
            assert_no_leak_paths(l, &adapter.spec().leak_globs)?;
            util::copy_tree(l, &dest.join("library"))?;
            lib = true;
        }
    }
    assert_no_leak_paths(dest, &adapter.spec().leak_globs)?;
    assert_no_grader_verdicts(dest, GRADER_VERDICT_MARKERS)?;
    if !rollouts.is_empty() && got_r.is_empty() {
        bail!(
            "no trajectory could be rendered for ANY rollout -- the proposer would be adapting on nothing: {}",
            failed_r.iter().take(4).cloned().collect::<Vec<_>>().join("; ")
        );
    }
    Ok(ContextTree {
        root: dest.to_path_buf(),
        task_id: task_id.into(),
        rollouts: got_r,
        deliverables: got_d,
        harness_files: n_h,
        library: lib,
        pool: got_p,
        failed: failed_r,
        deliverables_mode: if names_only { "names" } else { "contents" }.into(),
        source_files,
        decisions: decisions.map(|d| d.len()).unwrap_or(0),
        brief,
    })
}

fn leak_match(rel: &str, glob: &str) -> bool {
    let g = util::lstrip_dot_slash(glob);
    util::fnmatch(rel, g) || util::fnmatch(rel, &format!("*/{g}"))
}

/// Copy the task's source material under `root/source`, minus answer-key paths.
pub fn copy_source(adapter: &dyn BenchmarkAdapter, task_id: &str, root: &Path) -> Result<usize> {
    let Some(src) = adapter.task_files(task_id) else { return Ok(0) };
    if !src.is_dir() {
        return Ok(0);
    }
    let dst = root.join("source");
    let mut n = 0;
    for f in util::files_sorted(&src) {
        if util::in_pycache(&f) {
            continue;
        }
        let rel = util::rel(&src, &f);
        if adapter.spec().leak_globs.iter().any(|g| leak_match(&rel, g)) {
            continue;
        }
        let t = dst.join(&rel);
        if let Some(p) = t.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::copy(&f, &t)?;
        n += 1;
    }
    if n > 0 {
        assert_no_leak_paths(&dst, &adapter.spec().leak_globs)?;
    }
    Ok(n)
}

fn err_name(e: &anyhow::Error) -> String {
    format!("{}: {}", "Error", e)
}

/// Write one rollout's trajectory and deliverables; return "" or the failure detail.
fn write_rollout(dest: &Path, r: &RunView, adapter: &dyn BenchmarkAdapter, trunc: usize, names_only: bool) -> Result<String> {
    let mut failure = String::new();
    let tpath = dest.join("rollouts").join(format!("{}.md", r.label));
    match adapter.trajectory(&r.run_dir, trunc) {
        Ok(t) => util::write(&tpath, t)?,
        Err(e) => {
            util::write(&tpath, format!("(trajectory unavailable: {})", err_name(&e)))?;
            failure = format!("{}: {}", r.label, err_name(&e));
        }
    }
    let mut per_file = BTreeMap::new();
    if names_only {
        write_listing(&dest.join("deliverables").join(&r.label), &r.label, &listing(adapter, &r.run_dir))?;
    } else if let Some(files) = adapter.deliverable_files(&r.run_dir) {
        per_file = files;
    }
    if per_file.is_empty() {
        if !names_only {
            let text = match adapter.deliverable_text(&r.run_dir) {
                Ok(t) if !t.is_empty() => t,
                Ok(_) => "(no deliverable produced)".into(),
                Err(e) => format!("(deliverable unavailable: {})", err_name(&e)),
            };
            util::write(&dest.join("deliverables").join(format!("{}.md", r.label)), text)?;
        }
        return Ok(failure);
    }
    let d = dest.join("deliverables").join(&r.label);
    std::fs::create_dir_all(&d)?;
    for img in adapter.deliverable_images(&r.run_dir).into_iter().take(8) {
        if let Some(name) = img.file_name() {
            let t = d.join("images").join(name);
            let _ = std::fs::create_dir_all(t.parent().unwrap());
            let _ = std::fs::copy(&img, &t);
        }
    }
    for (name, body) in &per_file {
        let mut safe = name.replace('/', "__");
        if safe.is_empty() {
            safe = "unnamed".into();
        }
        let suffix = if safe.ends_with(".txt") { "" } else { ".txt" };
        util::write(&d.join(format!("{safe}{suffix}")), body)?;
    }
    let mut raw_names = vec![];
    for src in adapter.deliverable_assets(&r.run_dir).unwrap_or_default() {
        if let Some(name) = src.file_name() {
            let t = d.join("raw").join(name);
            std::fs::create_dir_all(t.parent().unwrap())?;
            std::fs::copy(&src, &t)?;
            raw_names.push(name.to_string_lossy().to_string());
        }
    }
    let raw_note = if raw_names.is_empty() {
        ""
    } else {
        "\n\n`raw/` holds the binary originals of the artifacts above (same names). The rendered\n`.txt` view is CANONICAL -- it reads each artifact the way the grader does (a formula\nwith no cached value renders as the empty cell the grader sees, even though the raw\nworkbook computes it on open). Open a raw file only for detail the rendering flattened,\nnever to overrule it.\n"
    };
    let lines: Vec<String> = per_file
        .iter()
        .map(|(n, b)| format!("- `{n}`  ({} chars extracted)", thousands(util::char_len(b))))
        .collect();
    util::write(
        &d.join("INDEX.md"),
        format!(
            "# {} produced {} file(s)\n\n{}\n\nEach `.txt` is the artifact converted to text by the SAME reader the work\nis assessed through, so reason about correctness from these -- an artifact\nthat does not survive that conversion is treated as unreadable however good\nit looks. `images/` holds any images from the same set, which are also part\nof how the work is read.\n{raw_note}",
            r.label,
            per_file.len(),
            lines.join("\n")
        ),
    )?;
    Ok(failure)
}

/// Python's `f"{n:,}"`.
pub fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn unified_diff(reference: &Tree, candidate: &Tree, label: &str) -> String {
    let mut names: Vec<&String> = reference.keys().chain(candidate.keys()).collect();
    names.sort();
    names.dedup();
    let mut out = String::new();
    for name in names {
        let a = reference.get(name).map(|s| s.as_str()).unwrap_or("");
        let b = candidate.get(name).map(|s| s.as_str()).unwrap_or("");
        if a == b {
            continue;
        }
        let d = similar::TextDiff::from_lines(a, b);
        out.push_str(
            &d.unified_diff()
                .context_radius(3)
                .header(&format!("current/{name}"), &format!("{label}/{name}"))
                .to_string(),
        );
    }
    out
}

/// Write each pool candidate's harness and rollout; return the labels written.
fn write_pool(dest: &Path, pool: &[PoolEntry], adapter: &dyn BenchmarkAdapter, trunc: usize, names_only: bool) -> Result<Vec<String>> {
    let mut got = vec![];
    for (label, cand_dir, roll) in pool {
        let d = dest.join("pool").join(label);
        std::fs::create_dir_all(&d)?;
        if cand_dir.is_dir() {
            assert_no_leak_paths(cand_dir, &adapter.spec().leak_globs)?;
            copy_component_tree(cand_dir, &d.join("harness"))?;
            let surface = adapter.surface();
            let reference = tree::read_tree(&surface, &dest.join("harness"), None);
            let candidate = tree::read_tree(&surface, cand_dir, None);
            util::write(&d.join("harness.diff"), unified_diff(&reference, &candidate, label))?;
        }
        if let Some(roll) = roll {
            let res: Result<()> = (|| {
                util::write(&d.join("rollout.md"), adapter.trajectory(&roll.run_dir, trunc)?)?;
                let produced = if names_only {
                    listing_md(label, &listing(adapter, &roll.run_dir))
                } else {
                    let t = adapter.deliverable_text(&roll.run_dir)?;
                    if t.is_empty() { "(nothing produced)".into() } else { t }
                };
                util::write(&d.join("produced.md"), produced)?;
                Ok(())
            })();
            if let Err(e) = res {
                util::write(&d.join("rollout.md"), format!("(unavailable: {e})"))?;
            }
        }
        got.push(label.clone());
    }
    Ok(got)
}

/// Deliverable names for one rollout, from the adapter, never a directory scan.
fn listing(adapter: &dyn BenchmarkAdapter, run_dir: &Path) -> Vec<String> {
    match adapter.deliverable_listing(run_dir) {
        None => vec!["(this adapter declares proposer_deliverables='names' but implements no deliverable_listing -- nothing can be listed)".into()],
        Some(Ok(v)) => v,
        Some(Err(e)) => vec![format!("(listing unavailable: {})", err_name(&e))],
    }
}

const WITHHELD_NOTE: &str = "CONTENTS ARE WITHHELD ON THIS PATH, DELIBERATELY. You are shown WHICH files the rollout\nproduced and not what is inside them. This is not an oversight and not a size budget:\non this benchmark the extracted-deliverable tree lives inside the same trial directory as\nthe grader's own output, so serving file bodies here once served the grader's pass rate\nand failed-check names along with them. Reason about the work from the TRAJECTORY -- what\nwas attempted, what was observed, what the run did with what it saw -- and from the file\nset itself, which is evidence in its own right: the task names the artifacts it expects,\nso a missing or misnamed file is a finding. Do not ask for the contents and do not infer\nthat an unlisted file is empty.\n";

fn listing_md(label: &str, names: &[String]) -> String {
    format!(
        "# {label} produced {} file(s)\n\n{label}: {}\n\n{}\n\n{WITHHELD_NOTE}",
        names.len(),
        tree::py_list(names),
        names.iter().map(|n| format!("- `{n}`")).collect::<Vec<_>>().join("\n")
    )
}

fn write_listing(d: &Path, label: &str, names: &[String]) -> Result<()> {
    std::fs::create_dir_all(d)?;
    util::write(&d.join("INDEX.md"), listing_md(label, names))
}

/// Copy a component tree, dropping byte-compiled files; return the file count.
pub fn copy_component_tree(src: &Path, dst: &Path) -> Result<usize> {
    let mut n = 0;
    for p in util::files_sorted(src) {
        if util::in_pycache(&p) || p.extension().map(|e| e == "pyc").unwrap_or(false) {
            continue;
        }
        let t = dst.join(util::rel(src, &p));
        std::fs::create_dir_all(t.parent().unwrap())?;
        std::fs::copy(&p, &t)?;
        n += 1;
    }
    Ok(n)
}

/// Reject benchmark-declared answer-key paths before a context reaches a model.
pub fn assert_no_leak_paths(root: &Path, patterns: &[String]) -> Result<()> {
    if patterns.is_empty() || !root.is_dir() {
        return Ok(());
    }
    let mut hits = vec![];
    for p in util::files_sorted(root) {
        let rel = util::rel(root, &p);
        if patterns.iter().any(|pat| leak_match(&rel, &pat.replace('\\', "/"))) {
            hits.push(rel);
        }
    }
    if !hits.is_empty() {
        hits.sort();
        bail!(
            "answer-key path(s) would enter model context: {}",
            hits.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

pub const GRADER_VERDICT_MARKERS: &[&str] = &[
    "deterministic_eval_summary",
    "deterministic_eval_result",
    "judge_scope_and_deterministic_status",
    "Judge findings on the rendered result",
    "\"passed_count\"",
    "\"pass_rate\"",
    "\"test_pass_rate\"",
];

const MARKER_SCAN_SUFFIXES: &[&str] = &[
    ".md", ".txt", ".json", ".csv", ".tsv", ".yaml", ".yml", ".html", ".htm", ".log", ".py", ".js",
    ".sh", ".xml", ".ini", ".toml", ".patch", ".diff", "",
];

fn suffix_of(rel: &str) -> String {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    match name.rfind('.') {
        Some(i) if i > 0 => name[i..].to_lowercase(),
        _ => String::new(),
    }
}

fn excerpt(body: &str, byte_at: usize) -> (usize, String) {
    let char_at = body[..byte_at].chars().count();
    let start = char_at.saturating_sub(60);
    let s: String = body.chars().skip(start).take(char_at - start + 120).collect();
    (char_at, format!("{s:?}"))
}

/// Refuse a context tree containing the grader's own output, by file name and by content.
pub fn assert_no_grader_verdicts(root: &Path, markers: &[&str]) -> Result<()> {
    if markers.is_empty() || !root.is_dir() {
        return Ok(());
    }
    for p in util::files_sorted(root) {
        let rel = util::rel(root, &p);
        for m in markers {
            if rel.contains(m) {
                bail!(
                    "grader verdict would enter model context: {rel:?} matched marker {m:?} in its NAME. The tripwire in core.context fired -- this is a leak, not a false positive to widen around. Find the read path that produced it."
                );
            }
        }
        if !MARKER_SCAN_SUFFIXES.contains(&suffix_of(&rel).as_str()) {
            continue;
        }
        let Ok(body) = util::read_lossy(&p) else { continue };
        for m in markers {
            if let Some(i) = body.find(m) {
                let (at, ex) = excerpt(&body, i);
                bail!(
                    "grader verdict would enter model context: {rel:?} contains marker {m:?} at offset {at} ({ex}). The tripwire in core.context fired -- this is a leak, not a false positive to widen around. Find the read path that produced it."
                );
            }
        }
    }
    Ok(())
}

/// The same tripwire on an agent-authored harness tree, before it can be rolled or stored.
pub fn assert_tree_has_no_grader_verdicts(tree: &Tree) -> Result<()> {
    for (rel, body) in tree {
        for m in GRADER_VERDICT_MARKERS {
            if rel.contains(m) {
                bail!(
                    "harness file {rel:?} has grader-verdict marker {m:?} in its NAME; rename it -- files carrying these strings cannot be served as evidence to later tasks"
                );
            }
        }
        if !MARKER_SCAN_SUFFIXES.contains(&suffix_of(rel).as_str()) {
            continue;
        }
        for m in GRADER_VERDICT_MARKERS {
            if let Some(i) = body.find(m) {
                let (at, ex) = excerpt(body, i);
                bail!(
                    "harness file {rel:?} contains grader-verdict marker {m:?} at offset {at} ({ex}); remove or rename that string -- files carrying it cannot be served as evidence to later tasks"
                );
            }
        }
    }
    Ok(())
}

/// The short pointer to the evidence on disk: the job, where the evidence is, and the
/// instruction to open it.
pub fn mission(tree: &ContextTree, job: &str) -> String {
    [
        job.trim().to_string(),
        String::new(),
        "# YOUR EVIDENCE".into(),
        String::new(),
        "It is on disk, in `context/`. Read it -- do not answer from this message alone.".into(),
        String::new(),
        tree.layout(),
        String::new(),
        "Every trajectory EVENT is present: the trajectory was not clipped to fit this mission. An individual observation may contain an explicit renderer-truncation marker; that marker is ours, not behavior by the solver. Do not diagnose it as a solver failure.".into(),
        String::new(),
        "Read `task.md` and the current harness first. Inventory every available rollout or candidate and inspect each at least once before deciding. For a candidate, compare what changed from its editing baseline, what behavior changed in its rollout, what improved, and what regressed or remained unresolved. Then deep-read the evidence relevant to any edit. Cite the specific trajectory moment behind every change; one you cannot trace to a moment is speculation.".into(),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_sep() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(1234), "1,234");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn verdict_tripwire() {
        let d = tempfile::tempdir().unwrap();
        util::write(&d.path().join("a.md"), "fine").unwrap();
        assert_no_grader_verdicts(d.path(), GRADER_VERDICT_MARKERS).unwrap();
        util::write(&d.path().join("b.json"), "{\"pass_rate\": 1}").unwrap();
        assert!(assert_no_grader_verdicts(d.path(), GRADER_VERDICT_MARKERS).is_err());
    }

    #[test]
    fn leak_globs() {
        let d = tempfile::tempdir().unwrap();
        util::write(&d.path().join("x/reference_answer.txt"), "k").unwrap();
        assert!(assert_no_leak_paths(d.path(), &["reference_answer.txt".into()]).is_err());
        assert!(assert_no_leak_paths(d.path(), &["other.txt".into()]).is_ok());
    }
}
