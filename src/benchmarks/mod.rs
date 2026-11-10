//! Benchmark adapters: the demo, open-world goals, and the three professional benchmarks
//! (JobBench, LAB, WorkBuddy-Bench), plus their shared helpers.

pub mod accuracy;
pub mod demo;
pub mod goal;
pub mod jb_oc_runner;
pub mod jobbench;
pub mod lab;
pub mod tasks;
pub mod wb_render;
pub mod workbuddy;

use crate::core::spec::{BenchPrompts, BenchmarkAdapter};
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Build BenchPrompts from extracted `BenchPrompts(...)` keyword data.
pub fn bp_from(v: &Value) -> BenchPrompts {
    let kw = v.get("__BenchPrompts__").unwrap_or(v);
    let mut bp = BenchPrompts::default();
    let g = |k: &str| kw.get(k).and_then(|x| x.as_str()).map(|s| s.to_string());
    if let Some(x) = g("work") { bp.work = x; }
    if let Some(x) = g("domain_word") { bp.domain_word = x; }
    if let Some(x) = g("inv_levers") { bp.inv_levers = x; }
    if let Some(x) = g("dom_levers") { bp.dom_levers = x; }
    if let Some(x) = g("yardstick") { bp.yardstick = x; }
    if let Some(x) = g("deliverable_word") { bp.deliverable_word = x; }
    if let Some(x) = g("domain_guidance") { bp.domain_guidance = x; }
    if let Some(x) = g("selection_checks") { bp.selection_checks = x; }
    if let Some(x) = g("selection_lead") { bp.selection_lead = x; }
    if let Some(x) = g("selection_conformance") { bp.selection_conformance = x; }
    if let Some(x) = g("prompt_pack") { bp.prompt_pack = x; }
    bp
}

/// Join non-empty parts with blank lines.
pub fn join(parts: &[&str]) -> String {
    parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n")
}

pub fn with_guidance(base: BenchPrompts, guidance: &[String]) -> BenchPrompts {
    let mut parts: Vec<&str> = vec![base.domain_guidance.as_str()];
    parts.extend(guidance.iter().map(|s| s.as_str()));
    let dg = join(&parts);
    BenchPrompts { domain_guidance: dg, ..base }
}

pub fn norm(value: &str) -> String {
    let a = Regex::new(r"[^a-z0-9]+").unwrap().replace_all(&value.to_lowercase(), "_").into_owned();
    Regex::new(r"_+").unwrap().replace_all(&a, "_").trim_matches('_').to_string()
}

pub fn lookup(table: &BTreeMap<String, String>, value: &str) -> String {
    if let Some(v) = table.get(value) {
        return v.clone();
    }
    let key = norm(value);
    table.iter().find(|(k, _)| norm(k) == key).map(|(_, v)| v.clone()).unwrap_or_default()
}

/// Run `argv` in its own process group; kill the group on a wall timeout. `Ok(None)` = timed out.
/// `log`: send stdout and stderr to this file (otherwise they are inherited).
pub fn run_detached(argv: &[String], env: &BTreeMap<String, String>, cwd: Option<&Path>, wall: u64, log: Option<&Path>) -> Result<Option<i32>> {
    use std::os::unix::process::CommandExt;
    use wait_timeout::ChildExt;
    let mut c = std::process::Command::new(&argv[0]);
    c.args(&argv[1..]).stdin(std::process::Stdio::null()).process_group(0);
    crate::config::apply_env(&mut c, env);
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    if let Some(l) = log {
        let f = std::fs::File::create(l)?;
        c.stdout(f.try_clone()?).stderr(f);
    }
    let mut child = c.spawn()?;
    match child.wait_timeout(std::time::Duration::from_secs(wall))? {
        Some(st) => Ok(Some(st.code().unwrap_or(-1))),
        None => {
            let _ = std::process::Command::new("kill").args(["-9", &format!("-{}", child.id())]).status();
            let _ = child.kill();
            let _ = child.wait();
            Ok(None)
        }
    }
}

/// The one place a CLI turns (--bench, --subset, --substrate) into an adapter.
pub fn make_adapter(bench: &str, subset: Option<&str>, substrate: &str, split_file: Option<&Path>) -> Result<Box<dyn BenchmarkAdapter>> {
    if bench == "goal" {
        let project = subset.unwrap_or("default");
        return Ok(Box::new(goal::GoalAdapter::open(&goal::project_dir(project)?)?));
    }
    if !["jb", "wb", "lab"].contains(&bench) {
        bail!("unknown bench {bench:?}; expected one of [\"jb\", \"wb\", \"lab\", \"goal\"]");
    }
    if !["oc", "cx"].contains(&substrate) {
        bail!("unknown substrate {substrate:?}; expected one of [\"oc\", \"cx\"]");
    }
    if bench == "wb" {
        let Some(s) = subset.filter(|s| ["office", "web", "code"].contains(s)) else {
            bail!("--bench wb needs --subset from [\"office\", \"web\", \"code\"], got {subset:?}");
        };
        return Ok(Box::new(workbuddy::WbAdapter::new(s, substrate, split_file)?));
    }
    if let Some(s) = subset {
        bail!("--subset applies to --bench wb only, got {s:?} for {bench}");
    }
    if bench == "jb" {
        return Ok(Box::new(jobbench::JobBenchAdapter::new(substrate, split_file)?));
    }
    Ok(Box::new(lab::LabAdapter::new(substrate, split_file)?))
}
