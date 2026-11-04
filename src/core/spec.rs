//! The benchmark seam: everything a benchmark must supply, and nothing more.
//!
//! Three axes stay separate -- method (recursion, prompts, selection, library), substrate (the
//! agent framework a solver rollout runs under) and benchmark (a `BenchSpec` value plus an
//! adapter). `ChannelSpec` describes the proposer/selector channel, which is independent of both.

use super::tree::{SurfaceSpec, Tree};
use crate::config;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub type ModelId = String;

/// `claude_cli` and `opencode` drive the reference CLIs; `api` is this port's native agent
/// runtime, which talks to a model endpoint directly (see `crate::llm`).
pub const CHANNELS: &[&str] = &["claude_cli", "opencode", "api"];

/// How to reach one model, on one proposer/selector channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelSpec {
    pub kind: String,
    pub model: ModelId,
    pub effort: String,
    pub steps: u32,
    pub timeout: u64,
}

impl ChannelSpec {
    pub fn new(kind: &str, model: &str) -> Result<ChannelSpec> {
        ChannelSpec::with(kind, model, "high", 120, 3600)
    }

    pub fn with(kind: &str, model: &str, effort: &str, steps: u32, timeout: u64) -> Result<ChannelSpec> {
        if !CHANNELS.contains(&kind) {
            bail!("unknown channel {kind:?}; expected one of {CHANNELS:?}");
        }
        Ok(ChannelSpec {
            kind: kind.into(),
            model: model.into(),
            effort: effort.into(),
            steps,
            timeout,
        })
    }
}

/// The one channel->model mapping; a lone override of either raises.
pub fn resolve_channel_model(
    kind: Option<&str>,
    model: Option<&str>,
    effort: &str,
    steps: u32,
    timeout: u64,
) -> Result<ChannelSpec> {
    let use_default = kind.is_none() && model.is_none();
    if kind.is_none() != model.is_none() {
        bail!(
            "channel and model must be overridden TOGETHER (a bare-id/provider-prefixed mismatch fails silently downstream); got kind={kind:?}, model={model:?}"
        );
    }
    let cfg = config::get();
    let kind = kind.map(|s| s.to_string()).unwrap_or_else(|| cfg.default_proposer_channel.clone());
    if !CHANNELS.contains(&kind.as_str()) {
        bail!("unknown channel {kind:?}; expected one of {CHANNELS:?}");
    }
    let model = model.map(|s| s.to_string()).unwrap_or_else(|| cfg.proposer_model_by_channel(&kind));
    if model.trim().is_empty() {
        if use_default {
            return ChannelSpec::with(&kind, "", effort, steps, timeout);
        }
        bail!("model override must not be empty");
    }
    if (kind == "opencode" || kind == "api") && !model.contains('/') {
        bail!(
            "{kind} channel needs a provider-prefixed model id in HARNESS2_IMPROVER_MODEL or HARNESS2_SOLVER_MODEL, got {model:?}"
        );
    }
    if kind == "claude_cli" && model.contains('/') {
        bail!("claude_cli channel takes a bare Anthropic id (e.g. \"claude-opus-4-1\"), got {model:?}");
    }
    ChannelSpec::with(&kind, &model, effort, steps, timeout)
}

/// The benchmark-shaped text the shared prompts interpolate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchPrompts {
    pub work: String,
    pub domain_word: String,
    pub inv_levers: String,
    pub dom_levers: String,
    pub yardstick: String,
    pub deliverable_word: String,
    pub domain_guidance: String,
    pub selection_checks: String,
    pub selection_lead: String,
    /// "full" | "value_only" | anything else (value tail only).
    pub selection_conformance: String,
    /// "adapt" | "tts".
    pub prompt_pack: String,
}

impl Default for BenchPrompts {
    fn default() -> Self {
        BenchPrompts {
            work: String::new(),
            domain_word: "domain".into(),
            inv_levers: String::new(),
            dom_levers: String::new(),
            yardstick: String::new(),
            deliverable_word: "deliverable".into(),
            domain_guidance: String::new(),
            selection_checks: String::new(),
            selection_lead: String::new(),
            selection_conformance: "full".into(),
            prompt_pack: "adapt".into(),
        }
    }
}

impl BenchPrompts {
    pub fn work(work: &str) -> BenchPrompts {
        BenchPrompts { work: work.into(), ..Default::default() }
    }

    /// The prompt fields must describe the WORK, never its measurement.
    pub fn checked(self) -> Result<BenchPrompts> {
        if self.prompt_pack != "adapt" && self.prompt_pack != "tts" {
            bail!("BenchPrompts.prompt_pack must be 'adapt' or 'tts', got {:?}", self.prompt_pack);
        }
        let visible = [
            ("work", &self.work),
            ("inv_levers", &self.inv_levers),
            ("dom_levers", &self.dom_levers),
            ("yardstick", &self.yardstick),
            ("domain_guidance", &self.domain_guidance),
            ("selection_checks", &self.selection_checks),
        ];
        for (field, value) in visible {
            let low = value.to_lowercase();
            for bad in ["benchmark", "evaluation", "grader", "rubric", "score"] {
                if low.contains(bad) {
                    bail!(
                        "BenchPrompts.{field} must describe the WORK, not its measurement (found {bad:?}). Say what the practitioner needs."
                    );
                }
            }
        }
        Ok(self)
    }
}

pub const INSTALL_MODES: &[&str] =
    &["workspace_agents_md", "home_dir_jinja", "workspace_config", "codex_root", "native"];

/// Static description of one (benchmark, substrate) cell. A value, not a code path.
#[derive(Debug, Clone)]
pub struct BenchSpec {
    pub name: String,
    pub runs_root: PathBuf,
    pub install_mode: String,
    pub prompts: BenchPrompts,
    pub channels: BTreeMap<String, ChannelSpec>,
    pub domain_word: String,
    pub leak_globs: Vec<String>,
    pub allow_subagents: bool,
    pub reserved_scripts: Vec<String>,
    /// "contents" | "names".
    pub proposer_deliverables: String,
    pub state_deliverable: bool,
    pub obs_trunc: usize,
}

impl BenchSpec {
    pub fn new(
        name: &str,
        runs_root: PathBuf,
        install_mode: &str,
        prompts: BenchPrompts,
        channels: BTreeMap<String, ChannelSpec>,
    ) -> BenchSpec {
        BenchSpec {
            name: name.into(),
            runs_root,
            install_mode: install_mode.into(),
            prompts,
            channels,
            domain_word: "domain".into(),
            leak_globs: vec![],
            allow_subagents: false,
            reserved_scripts: vec![],
            proposer_deliverables: "contents".into(),
            state_deliverable: false,
            obs_trunc: 50_000,
        }
    }

    pub fn checked(self) -> Result<BenchSpec> {
        if !INSTALL_MODES.contains(&self.install_mode.as_str()) {
            bail!("unknown install_mode {:?}; expected one of {INSTALL_MODES:?}", self.install_mode);
        }
        if self.proposer_deliverables != "contents" && self.proposer_deliverables != "names" {
            bail!(
                "BenchSpec.proposer_deliverables must be 'contents' or 'names', got {:?}",
                self.proposer_deliverables
            );
        }
        for role in ["proposer", "selector"] {
            if !self.channels.contains_key(role) {
                bail!("BenchSpec.channels must define {role:?}");
            }
        }
        if self.obs_trunc < 10_000 {
            bail!(
                "obs_trunc={} is in the range that fabricates defects; use >= 10_000 (default 50_000)",
                self.obs_trunc
            );
        }
        let mut bad: Vec<&String> =
            self.reserved_scripts.iter().filter(|s| s.contains('/') || !s.ends_with(".py")).collect();
        bad.sort();
        if !bad.is_empty() {
            bail!("BenchSpec.reserved_scripts takes bare .py BASENAMES, not paths: {bad:?}");
        }
        Ok(self)
    }
}

/// What the method knows about one finished rollout. Never carries a score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunView {
    pub label: String,
    pub run_id: String,
    pub run_dir: PathBuf,
    pub ok: bool,
    pub delivered: bool,
    pub cost_usd: f64,
    pub wall_s: f64,
    pub error: String,
}

impl RunView {
    pub fn new(label: &str, run_id: &str, run_dir: PathBuf, ok: bool, delivered: bool, error: &str) -> RunView {
        RunView {
            label: label.into(),
            run_id: run_id.into(),
            run_dir,
            ok,
            delivered,
            cost_usd: 0.0,
            wall_s: 0.0,
            error: error.into(),
        }
    }
}

/// Implement these and the whole method works. Nothing here is method logic. Methods with a
/// default body are the optional hooks of the Python protocol (`getattr(adapter, ...)`).
pub trait BenchmarkAdapter: Send + Sync {
    fn spec(&self) -> &BenchSpec;
    /// The substrate's short name ("oc" | "cx" | "nx"); it enters every namespace.
    fn substrate_name(&self) -> &str;
    /// The editable c1..c8 tree schema + validators for this substrate.
    fn surface(&self) -> SurfaceSpec;

    /// Task ids for a split. Stable order -- library state must be reproducible.
    fn tasks(&self, split: &str) -> Result<Vec<String>>;
    /// The grouping unit: practice area / occupation / role / sector.
    fn domain_of(&self, task_id: &str) -> String;
    fn reference_domain_order(&self) -> Option<Vec<String>> {
        None
    }
    fn prompts_for(&self, _task_id: &str) -> BenchPrompts {
        self.spec().prompts.clone()
    }
    /// The task as the agent received it, with criteria and expected output stripped.
    fn task_prompt(&self, task_id: &str) -> Result<String>;
    /// Optional. The source material the work was done against.
    fn task_files(&self, _task_id: &str) -> Option<PathBuf> {
        None
    }

    /// Run one rollout and return its run directory, idempotently by run_id.
    /// `harness_dir=None` means the unmodified base harness.
    fn execute(&self, task_id: &str, harness_dir: Option<&Path>, run_id: &str, timeout: Option<u64>) -> Result<PathBuf>;
    /// Optional native batching of base rollouts. `None` = not supported.
    fn execute_many(&self, _task_id: &str, _harness_dir: Option<&Path>, _run_ids: &[String]) -> Option<Result<Vec<PathBuf>>> {
        None
    }

    /// Did this rollout produce the artifact the task asked for?
    fn delivered(&self, run_dir: &Path) -> bool;
    /// Readable trajectory in the shared [ACTION]/[OBSERVATION]/[ERROR] grammar.
    fn trajectory(&self, run_dir: &Path, obs_trunc: usize) -> Result<String>;
    /// Optional. filename -> extracted readable text, one entry per artifact produced.
    fn deliverable_files(&self, _run_dir: &Path) -> Option<BTreeMap<String, String>> {
        None
    }
    /// Optional. Deliverable names only (for `proposer_deliverables = "names"`).
    fn deliverable_listing(&self, _run_dir: &Path) -> Option<Result<Vec<String>>> {
        None
    }
    fn deliverable_images(&self, _run_dir: &Path) -> Vec<PathBuf> {
        vec![]
    }
    /// Optional. Binary originals / rendered evidence to copy next to a candidate.
    fn deliverable_assets(&self, _run_dir: &Path) -> Option<Vec<PathBuf>> {
        None
    }
    fn deliverable_assets_note(&self, _names: &[String]) -> Option<String> {
        None
    }
    fn deliverable_assets_line(&self, _shown: &str) -> Option<String> {
        None
    }
    fn deliverable_assets_reference_only(&self) -> bool {
        false
    }
    /// The artifact as text, extracted from agent-authored bytes only.
    fn deliverable_text(&self, run_dir: &Path) -> Result<String>;
    /// Optional. `Some(false)` when the stored artifacts no longer match their record.
    fn artifacts_intact(&self, _run_dir: &Path) -> Option<bool> {
        None
    }

    /// A directory holding the unmodified c1..c8 tree this benchmark starts from.
    fn base_harness(&self) -> Result<PathBuf>;
    /// Benchmark-specific harness invariants. Fail listing all problems in actionable prose:
    /// the message is fed back verbatim as retry feedback.
    fn validate_tree(&self, tree: &Tree) -> Result<()>;
    /// Judge one run. `None` means the judge failed, which is not a zero and is never cached
    /// as one. Called only by `core::judge`, only after selection committed.
    fn score(&self, _run_dir: &Path, _task_id: &str) -> Result<Option<Value>> {
        Ok(None)
    }
    /// Health checks before money is spent.
    fn preflight(&self) -> Result<()> {
        Ok(())
    }
    /// Extra environment-specific readiness problems (used by `check`).
    fn readiness(&self, _judge: bool) -> Vec<String> {
        vec![]
    }
}

/// The task-specific prompt profile when the adapter supplies one.
pub fn resolve_prompts(adapter: &dyn BenchmarkAdapter, task_id: &str) -> BenchPrompts {
    adapter.prompts_for(task_id)
}

/// Helper for adapters that want a `{proposer, selector}` channel map.
pub fn channel_map(proposer: ChannelSpec, selector: ChannelSpec) -> BTreeMap<String, ChannelSpec> {
    let mut m = BTreeMap::new();
    m.insert("proposer".into(), proposer);
    m.insert("selector".into(), selector);
    m
}

pub type Env = HashMap<String, String>;
