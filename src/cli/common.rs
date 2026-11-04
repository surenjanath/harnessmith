//! Shared plumbing for the CLIs: adapter construction, the on-disk conventions, and `run`.

use crate::benchmarks;
use crate::core::library;
use crate::core::proposer::{Propose, Proposer};
use crate::core::schedule::{self, RunOpts, SweepOpts};
use crate::core::spec::BenchmarkAdapter;
use crate::{config, util};
use anyhow::{bail, Result};
use clap::Args;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const MODES_HELP: &str = "parallel | sequential (agg / seq accepted: the on-disk tokens); comma list, e.g. 'parallel' or 'parallel,sequential'";

#[derive(Args, Clone, Debug)]
pub struct CellArgs {
    /// jb | wb | lab | goal
    #[arg(long)]
    pub bench: String,
    /// WB only: office | web | code. For --bench goal: the project name.
    #[arg(long)]
    pub subset: Option<String>,
    /// task-executor substrate: oc (opencode) | cx (codex)
    #[arg(long, default_value = "oc")]
    pub substrate: String,
}

#[derive(Args, Clone, Debug)]
pub struct IdentityArgs {
    /// scale: R=4k rollouts, P=3k improver sessions per task
    #[arg(long, default_value_t = 1)]
    pub k: usize,
    /// stream-order seed; omit for sorted task order
    #[arg(long)]
    pub seed: Option<i64>,
    #[arg(long = "mode", alias = "modes", default_value = "parallel", help = MODES_HELP)]
    pub modes: String,
}

#[derive(Args, Clone, Debug)]
pub struct RunArgs {
    #[command(flatten)]
    pub cell: CellArgs,
    #[command(flatten)]
    pub id: IdentityArgs,
    #[arg(long, default_value = "all")]
    pub split: String,
    /// optional partition from scripts/split_tasks.py
    #[arg(long)]
    pub split_file: Option<PathBuf>,
    /// domains to run concurrently
    #[arg(long, default_value_t = 1)]
    pub workers: usize,
    /// comma list restricting the sweep to these domains
    #[arg(long, default_value = "")]
    pub only_domains: String,
    /// per-domain cap on the stream (the first N tasks of every domain)
    #[arg(long)]
    pub max_tasks: Option<usize>,
    /// print the stream order, the namespaces and the split's domains, then exit
    #[arg(long)]
    pub dry_run: bool,
}

pub fn make_adapter(cell: &CellArgs, split_file: Option<&Path>) -> Result<Box<dyn BenchmarkAdapter>> {
    benchmarks::make_adapter(&cell.bench, cell.subset.as_deref(), &cell.substrate, split_file)
}

/// Scheduler state: summaries, harnesses and selections, not the rollout store.
pub fn runs_root_for(adapter: &dyn BenchmarkAdapter) -> PathBuf {
    adapter.spec().runs_root.join("adapt")
}

/// Per-(benchmark, substrate, k, seed, tag) experience-library root.
pub fn library_root_for(adapter: &dyn BenchmarkAdapter, k: usize, seed: Option<i64>, tag: &str) -> Result<PathBuf> {
    schedule::validate_tag(tag)?;
    let mut variant = adapter.substrate_name().to_string();
    if let Some(s) = seed {
        variant += &format!("-s{s}");
    }
    if !tag.is_empty() {
        variant += &format!("-{tag}");
    }
    Ok(library::root_for(&adapter.spec().runs_root.join("library"), &adapter.spec().name, k, &variant, ""))
}

/// The namespace whose task dirs hold summary.json for BOTH modes.
pub fn summary_namespace(adapter: &dyn BenchmarkAdapter, k: usize, seed: Option<i64>, tag: &str) -> Result<String> {
    schedule::ns(&adapter.spec().name, adapter.substrate_name(), "agg", k, seed, tag)
}

/// 'parallel,sequential' (or 'agg,seq') -> tokens in MODES order.
pub fn parse_modes(raw: &str) -> Result<Vec<&'static str>> {
    let names: Vec<&str> = raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if names.is_empty() {
        bail!("--modes takes a comma list of parallel | sequential (agg | seq), got {raw:?}");
    }
    let mut toks = vec![];
    for n in names {
        toks.push(schedule::mode_token(n).map_err(|e| anyhow::anyhow!("--modes: {e} (got {raw:?})"))?);
    }
    Ok(schedule::MODES.iter().filter(|m| toks.contains(m)).cloned().collect())
}

pub fn csv(raw: &str) -> Vec<String> {
    raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

fn stream_for(adapter: &dyn BenchmarkAdapter, a: &RunArgs) -> Result<(Vec<(String, Vec<String>)>, Vec<(String, String)>)> {
    let by = schedule::task_stream(adapter, &a.split, a.id.seed, &csv(&a.only_domains), &[], a.max_tasks)?;
    let flat = by.iter().flat_map(|(d, ts)| ts.iter().map(move |t| (d.clone(), t.clone()))).collect();
    Ok((by, flat))
}

fn print_stream(adapter: &dyn BenchmarkAdapter, a: &RunArgs, modes: &[&str]) -> Result<i32> {
    let (by, stream) = stream_for(adapter, a)?;
    let labels: Vec<&str> = modes.iter().map(|m| schedule::mode_name(m)).collect();
    println!(
        "# {}-{} split={} seed={} k={} modes={}",
        adapter.spec().name,
        adapter.substrate_name(),
        a.split,
        a.id.seed.map(|s| s.to_string()).unwrap_or_else(|| "None".into()),
        a.id.k,
        labels.join(",")
    );
    for m in modes {
        println!("#   namespace[{}]: {}", schedule::mode_name(m), schedule::ns(&adapter.spec().name, adapter.substrate_name(), m, a.id.k, a.id.seed, "")?);
    }
    println!("#   summaries under: {}/{}", runs_root_for(adapter).display(), summary_namespace(adapter, a.id.k, a.id.seed, "")?);
    println!("#   library root:    {}", library_root_for(adapter, a.id.k, a.id.seed, "")?.display());
    println!("#   domains parallel: {}", a.workers);
    for (i, (d, t)) in stream.iter().enumerate() {
        println!("{:>4}  {d:<40} {t}", i + 1);
    }
    println!("# domains of this split, with task counts (the legal --only-domains names):");
    for (d, ts) in &by {
        println!("{:>4}  {d}", ts.len());
    }
    let cap = a.max_tasks.map(|m| format!(", max {m} per domain")).unwrap_or_default();
    println!("# {} tasks over {} domains (domains parallel, tasks within a domain sequential{cap})", stream.len(), by.len());
    Ok(if stream.is_empty() { 1 } else { 0 })
}

/// Prevent a resumed run or library from silently mixing models or task partitions.
fn check_identity(adapter: &dyn BenchmarkAdapter, a: &RunArgs, ns_dir: &Path) -> Result<()> {
    let cfg = config::get();
    let mut domains = csv(&a.only_domains);
    domains.sort();
    let identity = json!({
        "method_version": config::METHOD_VERSION,
        "executor": if a.cell.substrate == "cx" { cfg.solver_model_cx.clone() } else { cfg.solver_model_oc.clone() },
        "improver": cfg.improver_model,
        "judge": cfg.judge_model,
        "project": cfg.vertex_project,
        "location": cfg.vertex_region,
        "judge_endpoint": cfg.jb_judge_api_base,
        "endpoint": if a.cell.substrate == "cx" { cfg.codex_base_url.clone() } else { String::new() },
        "split": a.split,
        "tasks": stream_for(adapter, a)?.1,
        "max_tasks": a.max_tasks,
        "domains": domains,
        "seed": a.id.seed,
    });
    std::fs::create_dir_all(ns_dir)?;
    let p = ns_dir.join("config.json");
    if p.is_file() {
        let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        if old != identity {
            bail!("This run directory belongs to different models or tasks. Choose a new HARNESS2_RUNS_DIR for a new experiment.");
        }
    } else {
        util::write(&p, format!("{}\n", serde_json::to_string_pretty(&identity)?))?;
    }
    Ok(())
}

pub fn run_sweep_cmd(a: RunArgs) -> Result<i32> {
    if a.id.k < 1 || a.workers < 1 {
        bail!("--k and --workers must be positive");
    }
    let modes = parse_modes(&a.id.modes)?;
    let adapter = make_adapter(&a.cell, a.split_file.as_deref())?;
    if a.dry_run {
        return print_stream(adapter.as_ref(), &a, &modes);
    }
    stream_for(adapter.as_ref(), &a)?;
    crate::cli::setup::require_ready(adapter.as_ref(), true)?;
    let runs_root = runs_root_for(adapter.as_ref());
    let library_root = library_root_for(adapter.as_ref(), a.id.k, a.id.seed, "")?;
    let summary_ns = summary_namespace(adapter.as_ref(), a.id.k, a.id.seed, "")?;
    if a.cell.bench != "goal" {
        check_identity(adapter.as_ref(), &a, &runs_root.join(&summary_ns))?;
    }
    let prop: Arc<dyn Propose> = Arc::new(Proposer::new(adapter.spec().channels["proposer"].clone(), Some(runs_root.join("_audit").join(&summary_ns)), 2)?);
    println!("sweep {summary_ns}: {} domain(s) in parallel, tasks within a domain in stream order", a.workers);
    let o = RunOpts { modes: modes.clone(), seed: a.id.seed, ..Default::default() };
    let s = SweepOpts { split: a.split.clone(), domain_parallel: a.workers, only_domains: csv(&a.only_domains), max_tasks: a.max_tasks, only_tasks: vec![] };
    let rep = schedule::run_sweep(a.id.k, adapter.as_ref(), &prop, &runs_root, &library_root, &o, &s)?;
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let rec = runs_root.join("_sweeps").join(format!("{summary_ns}-{stamp}.json"));
    util::write(&rec, serde_json::to_string_pretty(&rep)?)?;
    let doms = rep["domains"].as_array().cloned().unwrap_or_default();
    let done: u64 = doms.iter().map(|d| d["done"].as_u64().unwrap_or(0)).sum();
    let failed: u64 = doms.iter().map(|d| d["failed"].as_u64().unwrap_or(0)).sum();
    let dead: Vec<String> = doms.iter().filter(|d| d.get("error").is_some()).map(|d| d["domain"].as_str().unwrap_or("").to_string()).collect();
    if doms.is_empty() {
        println!(
            "\nsweep {summary_ns}: zero tasks scheduled (split={:?}, only_domains={:?}, max_tasks={:?}); refusing to report success",
            a.split, csv(&a.only_domains), a.max_tasks
        );
        println!("record: {}", rec.display());
        return Ok(1);
    }
    println!(
        "\nsweep {summary_ns}: {done} tasks ok, {failed} failed, {} domain(s) died{}",
        dead.len(),
        if dead.is_empty() { String::new() } else { format!(": {}", dead.join(", ")) }
    );
    println!("record: {}", rec.display());
    println!(
        "next:   harness2 judge --bench {}{} --substrate {} --k {}{} --modes {}",
        a.cell.bench,
        a.cell.subset.as_ref().map(|s| format!(" --subset {s}")).unwrap_or_default(),
        a.cell.substrate,
        a.id.k,
        a.id.seed.map(|s| format!(" --seed {s}")).unwrap_or_default(),
        a.id.modes
    );
    Ok(if !dead.is_empty() { 1 } else if failed > 0 { 2 } else { 0 })
}
