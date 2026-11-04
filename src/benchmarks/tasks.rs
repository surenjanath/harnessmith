//! Task discovery and optional user-created partitions; no bundled task lists.

use crate::{config, util};
use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Children of `root` (sorted) that are directories.
fn subdirs(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
    v.sort();
    v
}

/// Sorted task IDs mapped to their dataset-provided domain.
pub fn discover(bench: &str, subset: Option<&str>) -> Result<BTreeMap<String, String>> {
    let cfg = config::get();
    let mut tasks = BTreeMap::new();
    let root: PathBuf;
    match (bench, subset) {
        ("jb", _) => {
            root = cfg.jb_repo().join("dataset/main");
            for a in subdirs(&root) {
                for b in subdirs(&a) {
                    if b.join("task_folder/TASK_INSTRUCTIONS.txt").is_file() {
                        let rel = util::rel(&root, &b);
                        let dom = rel.split('/').next().unwrap_or("").to_string();
                        tasks.insert(rel, dom);
                    }
                }
            }
        }
        ("lab", _) => {
            root = cfg.lab_repo().join("tasks");
            for a in subdirs(&root) {
                for b in subdirs(&a) {
                    if b.join("task.json").is_file() {
                        let rel = util::rel(&root, &b);
                        let dom = rel.split('/').next().unwrap_or("").to_string();
                        tasks.insert(rel, dom);
                    }
                }
            }
        }
        ("wb", Some(s)) if ["office", "web", "code"].contains(&s) => {
            root = cfg.wb_repo().join(format!("datasets/wb-bench-{s}-v1.0/tasks"));
            let field = if s == "code" { "role" } else { "category" };
            for d in subdirs(&root) {
                let p = d.join("task.toml");
                if !p.is_file() {
                    continue;
                }
                let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&p)?)?;
                let domain = doc.get("metadata").and_then(|m| m.get(field)).and_then(|v| v.as_str()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                let Some(domain) = domain else { bail!("{}: metadata.{field} must be a nonempty string", p.display()) };
                if !d.join("instruction.md").is_file() {
                    bail!("Missing task instructions: {}", d.join("instruction.md").display());
                }
                tasks.insert(d.file_name().unwrap().to_string_lossy().to_string(), domain);
            }
        }
        _ => bail!("Unknown benchmark/subset: {bench}/{}", subset.unwrap_or("None")),
    }
    if tasks.is_empty() {
        bail!(
            "No {bench} tasks found at {}. Run `harness2 setup {bench}{}`.",
            root.display(),
            subset.map(|s| format!(" --subset {s}")).unwrap_or_default()
        );
    }
    Ok(tasks)
}

pub fn select(tasks: &BTreeMap<String, String>, split: &str, path: Option<&Path>, benchmark: &str) -> Result<Vec<String>> {
    if !["all", "train", "test"].contains(&split) {
        bail!("Unknown split {split:?}; use all, train, or test");
    }
    let Some(path) = path else {
        if split != "all" {
            bail!("--split train/test requires --split-file; create one with python scripts/split_tasks.py, or use --split all");
        }
        return Ok(tasks.keys().cloned().collect());
    };
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if data.get("benchmark").and_then(|b| b.as_str()) != Some(benchmark) {
        bail!("{}: expected a split for {benchmark}", path.display());
    }
    let mut halves: Vec<HashSet<String>> = vec![];
    for name in ["train", "test"] {
        let Some(ids) = data.get(name).and_then(|v| v.as_array()) else { bail!("{}: {name} must be a list of task IDs", path.display()) };
        let ids: Vec<String> = ids.iter().map(|v| v.as_str().map(|s| s.to_string())).collect::<Option<_>>().ok_or_else(|| anyhow::anyhow!("{}: {name} must be a list of task IDs", path.display()))?;
        let set: HashSet<String> = ids.iter().cloned().collect();
        if set.len() != ids.len() {
            bail!("{}: duplicate task IDs in {name}", path.display());
        }
        halves.push(set);
    }
    if !halves[0].is_disjoint(&halves[1]) {
        bail!("{}: train and test overlap", path.display());
    }
    let mut unknown: Vec<&String> = halves[0].union(&halves[1]).filter(|t| !tasks.contains_key(*t)).collect();
    unknown.sort();
    if !unknown.is_empty() {
        bail!("{}: task IDs absent from the dataset: {:?}", path.display(), &unknown[..unknown.len().min(5)]);
    }
    let mut out: Vec<String> = match split {
        "all" => halves[0].union(&halves[1]).cloned().collect(),
        "train" => halves[0].iter().cloned().collect(),
        _ => halves[1].iter().cloned().collect(),
    };
    out.sort();
    Ok(out)
}
