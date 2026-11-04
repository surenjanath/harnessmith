//! Command-line entry points. One binary, `harness2`, with the reference CLIs as subcommands:
//!
//! ```text
//!     harness2 run      (= harness2)          adaptation sweep over a benchmark
//!     harness2 judge    (= harness2-judge)    grade the selected runs
//!     harness2 report   (= harness2-report)   summarise judged runs
//!     harness2 setup    (= harness2-setup)    install a benchmark's pinned dependencies
//!     harness2 check    (= harness2-check)    preflight a benchmark cell
//!     harness2 demo     (= harness2-demo)     deterministic local demo, no credentials
//!     harness2 bridge   (= harness2-bridge)   Vertex AI proxy for Codex / WorkBuddy
//!     harness2 goal                           refine a harness for one goal you type
//!     harness2 models                         manage model connections
//!     harness2 serve                          the web UI
//! ```

pub mod common;
pub mod judge;
pub mod setup;

use crate::benchmarks::demo::{DemoAdapter, DemoProposer};
use crate::benchmarks::goal::{self, GoalAdapter};
use crate::core::proposer::{Propose, Proposer};
use crate::core::schedule::{self, RunOpts};
use crate::core::spec::BenchmarkAdapter;
use crate::llm::{client, connections};
use crate::{config, util};
use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "harness2", version, about = "Harness²: Recursive Agent Harnessing for an Open World (Rust port)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run one adaptation sweep: every task of a split, domains parallel, tasks in stream order.
    Run(common::RunArgs),
    /// Judge the run that selection chose, per task and mode.
    Judge(judge::JudgeArgs),
    /// Print the benchmark figure for judged selections.
    Report(judge::ReportArgs),
    /// Install a benchmark's pinned dependencies (and OpenCode / Codex).
    Setup(setup::SetupArgs),
    /// Check that a benchmark cell is ready to run, without spending anything.
    Check(setup::CheckArgs),
    /// Run the real recursion and library workflow with local, deterministic editors.
    Demo(DemoArgs),
    /// Serve a local OpenAI-compatible proxy in front of Vertex AI (for Codex / WorkBuddy).
    Bridge(setup::BridgeArgs),
    /// Refine a harness for a goal you type, then deliver the selected output.
    Goal(GoalArgs),
    /// Manage model connections (list, add, test, remove).
    Models(ModelsArgs),
    /// Start the web UI.
    Serve(ServeArgs),
    /// (internal) the JobBench OpenCode launcher; reads its settings from the environment.
    #[command(hide = true, name = "jb-oc-runner")]
    JbOcRunner,
}

#[derive(Args)]
struct DemoArgs {
    #[arg(long, default_value = "runs/demo")]
    output: PathBuf,
    #[arg(long, default_value_t = 2)]
    k: usize,
    /// parallel | sequential | both
    #[arg(long, default_value = "both")]
    mode: String,
}

#[derive(Args)]
pub struct GoalArgs {
    /// The goal / task, in plain language.
    pub goal: String,
    /// Project: goals in one project share an experience library.
    #[arg(long, default_value = "default")]
    pub project: String,
    /// Input files to give the agent.
    #[arg(long = "file")]
    pub files: Vec<PathBuf>,
    #[arg(long, default_value = "general")]
    pub domain: String,
    #[arg(long, default_value_t = 1)]
    pub k: usize,
    /// parallel | sequential | both
    #[arg(long, default_value = "parallel")]
    pub mode: String,
    /// `<connection>/<model>` for executing the task.
    #[arg(long)]
    pub solver: Option<String>,
    /// `<connection>/<model>` for editing the harness.
    #[arg(long)]
    pub improver: Option<String>,
    /// `<connection>/<model>` for choosing among final outputs.
    #[arg(long)]
    pub selector: Option<String>,
    /// Grading criteria (one per line) for an optional post-hoc judge; never shown to models.
    #[arg(long, default_value = "")]
    pub rubric: String,
}

#[derive(Args)]
struct ModelsArgs {
    #[command(subcommand)]
    cmd: ModelsCmd,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List connections and role defaults.
    List,
    /// Add or replace a connection.
    Add {
        name: String,
        /// openai | anthropic | gemini | ollama | mock
        #[arg(long)]
        provider: String,
        #[arg(long, default_value = "")]
        base_url: String,
        /// Environment variable that holds the key.
        #[arg(long, default_value = "")]
        api_key_env: String,
        #[arg(long, default_value = "")]
        default_model: String,
    },
    /// Remove a connection.
    Remove { name: String },
    /// Send a one-line request through `<connection>/<model>`.
    Test { model: String },
    /// Set the default model for a role: solver | improver | selector | judge.
    Role { role: String, model: String },
}

#[derive(Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8765)]
    port: u16,
}

pub fn main() -> i32 {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Run(a) => common::run_sweep_cmd(a),
        Cmd::Judge(a) => judge::judge_cmd(a),
        Cmd::Report(a) => judge::report_cmd(a),
        Cmd::Setup(a) => setup::setup_cmd(a),
        Cmd::Check(a) => setup::check_cmd(a),
        Cmd::Demo(a) => demo(a),
        Cmd::Bridge(a) => setup::bridge_cmd(a),
        Cmd::Goal(a) => goal_cmd(a).map(|_| 0),
        Cmd::Models(a) => models(a).map(|_| 0),
        Cmd::Serve(a) => crate::ui::serve(&a.host, a.port).map(|_| 0),
        Cmd::JbOcRunner => crate::benchmarks::jb_oc_runner::main(),
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

fn demo(a: DemoArgs) -> Result<i32> {
    if a.k < 1 {
        bail!("--k must be positive");
    }
    std::fs::create_dir_all(&a.output)?;
    let root = std::fs::canonicalize(&a.output)?;
    let adapter = DemoAdapter::new(&root)?;
    let prop: Arc<dyn Propose> = Arc::new(DemoProposer::new(adapter.spec().channels["proposer"].clone()));
    let modes: Vec<&'static str> = if a.mode == "both" { schedule::MODES.to_vec() } else { vec![schedule::mode_token(&a.mode)?] };
    let o = RunOpts { modes, ..Default::default() };
    let tasks = adapter.tasks("all")?;
    let mut result = schedule::run_domain("accounting", &tasks, a.k, &adapter, &prop, &adapter.spec().runs_root.clone(), &root.join("library").join(format!("k{}", a.k)), &o);
    result["execution_count"] = json!(adapter.execution_count.load(std::sync::atomic::Ordering::SeqCst));
    result["editor_calls"] = json!(prop.calls().len());
    let report = root.join("demo.json");
    util::write(&report, format!("{}\n", serde_json::to_string_pretty(&result)?))?;
    println!(
        "Local demo: {} tasks completed; {} failed. {}",
        result["done"], result["failed"], report.display()
    );
    Ok(if result["failed"].as_u64().unwrap_or(0) > 0 { 1 } else { 0 })
}

/// Build the proposer for a goal adapter's channel.
pub fn goal_proposer(adapter: &GoalAdapter, audit: PathBuf) -> Result<Arc<dyn Propose>> {
    Ok(Arc::new(Proposer::new(adapter.spec().channels["proposer"].clone(), Some(audit), 2)?))
}

pub fn goal_cmd(a: GoalArgs) -> Result<serde_json::Value> {
    let root = goal::project_dir(&a.project)?;
    let mut adapter = GoalAdapter::open(&root)?;
    if let Some(s) = a.solver {
        adapter.settings.solver_model = s;
    }
    if let Some(s) = a.improver {
        adapter.settings.improver_model = s;
    }
    if let Some(s) = a.selector {
        adapter.settings.selector_model = s;
    }
    if adapter.settings.improver_model.is_empty() {
        adapter.settings.improver_model = adapter.settings.solver_model.clone();
    }
    if adapter.settings.selector_model.is_empty() {
        adapter.settings.selector_model = adapter.settings.improver_model.clone();
    }
    let adapter = GoalAdapter::with_settings(&root, adapter.settings.clone())?;
    adapter.save_settings()?;
    adapter.preflight()?;
    let mut files = vec![];
    for f in &a.files {
        files.push((f.file_name().unwrap().to_string_lossy().to_string(), std::fs::read(f)?));
    }
    let id = adapter.add_task(&a.goal, &a.domain, &a.rubric, &files)?;
    let modes: Vec<&'static str> = if a.mode == "both" { schedule::MODES.to_vec() } else { vec![schedule::mode_token(&a.mode)?] };
    run_goal(&adapter, &id, a.k, modes)
}

/// Run every mode for one registered goal, threading the project library.
pub fn run_goal(adapter: &GoalAdapter, id: &str, k: usize, modes: Vec<&'static str>) -> Result<serde_json::Value> {
    let runs_root = common::runs_root_for(adapter);
    let lib_root = common::library_root_for(adapter, k, None, "")?;
    let prop = goal_proposer(adapter, runs_root.join("_audit"))?;
    let o = RunOpts { modes: modes.clone(), width: adapter.settings.parallelism.max(1), ..Default::default() };
    let domain = adapter.domain_of(id);
    let surface = adapter.surface();
    let libs: std::collections::BTreeMap<String, crate::core::library::Library> = modes
        .iter()
        .map(|m| (m.to_string(), crate::core::library::Library::new(lib_root.join(m), &domain, surface.clone())))
        .collect();
    let res = schedule::run_task(id, k, adapter, &prop, &runs_root, &libs, &o)?;
    let mut out = res.to_json();
    for m in &modes {
        if let Some(rd) = res.selections.get(*m).and_then(|s| s.get("chosen_run_dir")).and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            let text = adapter.deliverable_text(std::path::Path::new(rd)).unwrap_or_default();
            out["deliverables"][*m] = json!({"run_dir": rd, "text": text});
            crate::say!("\n===== {} result ({}) =====\n{}", schedule::mode_name(m), rd, text);
        }
    }
    Ok(out)
}

fn models(a: ModelsArgs) -> Result<()> {
    let mut reg = connections::load();
    match a.cmd {
        ModelsCmd::List => {
            println!("connections file: {}", connections::path().display());
            for c in &reg.connections {
                println!(
                    "  {:<16} {:<10} {:<40} key:{} default:{}",
                    c.name,
                    c.provider,
                    c.base(),
                    if c.key().is_empty() { "none" } else { "set" },
                    c.default_model
                );
            }
            for (r, m) in &reg.roles {
                println!("  role {r:<9} -> {m}");
            }
            println!("  built-in prefixes: openai anthropic gemini google ollama openrouter groq deepseek together mistral mock");
        }
        ModelsCmd::Add { name, provider, base_url, api_key_env, default_model } => {
            reg.connections.retain(|c| c.name != name);
            reg.connections.push(connections::Connection {
                name,
                provider,
                base_url,
                api_key: String::new(),
                api_key_env,
                default_model,
                input_price: 0.0,
                output_price: 0.0,
                max_tokens: None,
                temperature: None,
                context_window: None,
                max_concurrency: None,
            });
            connections::save(&reg)?;
            println!("saved {}", connections::path().display());
        }
        ModelsCmd::Remove { name } => {
            reg.connections.retain(|c| c.name != name);
            connections::save(&reg)?;
        }
        ModelsCmd::Test { model } => {
            let (c, m) = connections::resolve(&model)?;
            println!("{} -> {}", model, client::ping(&c, &m)?);
        }
        ModelsCmd::Role { role, model } => {
            if !["solver", "improver", "selector", "judge"].contains(&role.as_str()) {
                bail!("role must be solver | improver | selector | judge");
            }
            reg.roles.insert(role, model);
            connections::save(&reg)?;
        }
    }
    let _ = config::get();
    Ok(())
}
