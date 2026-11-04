//! JobBench adapter: tasks, splits, prompts, launcher bodies, deliverables and scores.
//!
//! Rollouts are addressed by LABEL, not by run directory: the run store is dataset-addressed
//! and shared by every setting, so the run_id becomes the label.

use super::{bp_from, lookup, run_detached, tasks, with_guidance};
use crate::cli::setup::codex_config_error;
use crate::config;
use crate::core::runmark as RM;
use crate::core::spec::{channel_map, resolve_channel_model, BenchPrompts, BenchSpec, BenchmarkAdapter};
use crate::core::tree::{self, SurfaceSpec, Tree};
use crate::substrate::codex::CodexSubstrate;
use crate::substrate::opencode::{OpencodeSubstrate, OUTPUT_TOKEN_MAX};
use crate::substrate::{Install, RunRequest};
use crate::{say, util};
use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

static DATA: Lazy<Value> = Lazy::new(|| serde_json::from_str(include_str!("data/jobbench.json")).unwrap());
static STORE_SEED_LOCK: Mutex<()> = Mutex::new(());
static EXTRACT_CACHE: Lazy<Mutex<HashMap<String, Value>>> = Lazy::new(|| Mutex::new(HashMap::new()));

const MAX_CHARS_PER_FILE: usize = 200_000;
const VISION_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp"];
const MAX_VISION: usize = 8;
const JUDGE_FAILURE_PREFIXES: &[&str] = &["opus judge failed", "failed after", "failed to get judge response", "judge raised an exception:"];

pub fn runner_dir() -> PathBuf {
    config::get().third_party.join("jb_runner")
}

fn table(name: &str) -> BTreeMap<String, String> {
    DATA[name].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()
}

/// JobBench-wide profile plus SOC-style domain and occupation overlays.
pub fn jobbench_prompts(domain: &str, profession: &str) -> BenchPrompts {
    let base = bp_from(&DATA["JOBBENCH_BASE"]);
    with_guidance(base, &[lookup(&table("JOBBENCH_DOMAINS"), domain), lookup(&table("JOBBENCH_PROFESSIONS"), profession)])
}

static TOKEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\{\{\s*(\w+)\s*\}\}").unwrap());
static COMMENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());

/// Substitute {{ name }} tokens only; single braces in prose stay untouched.
pub fn interp(text: &str, ctx: &BTreeMap<String, String>) -> String {
    TOKEN.replace_all(text, |c: &regex::Captures| ctx.get(&c[1]).cloned().unwrap_or_default()).into_owned()
}

/// Drop HTML comment blocks, so a comment-only component renders to nothing.
pub fn strip_comments(text: &str) -> String {
    COMMENT.replace_all(text, "").trim().to_string()
}

fn read_component(staged: &Path, rel: &str) -> String {
    util::read_lossy(&staged.join(rel)).unwrap_or_default()
}

pub fn render_ctx(domain: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("env_description".into(), "A command-line workspace containing the task folder and an output directory; standard CLI + Python tooling is available.".into());
    m.insert("domain_focus".into(), domain.into());
    m.insert("date".into(), chrono::Utc::now().date_naive().to_string());
    m.insert("working_directory".into(), "the temp workspace (task_folder/ has inputs; write to output/)".into());
    m
}

fn list_scripts(staged: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(staged.join("scripts"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().map(|e| e == "py").unwrap_or(false) && !p.file_name().unwrap().to_string_lossy().starts_with('_'))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn script_doc(p: &Path) -> String {
    let Ok(txt) = util::read_lossy(p) else { return "helper script".into() };
    Regex::new(r#"(?m)^\s*(?:"""|''')(.+)"#)
        .unwrap()
        .captures(&txt)
        .map(|c| util::take_chars(c[1].trim(), 120))
        .unwrap_or_else(|| "helper script".into())
}

/// c1 prose + c6 guardrails + c4 tool guidance + c3 memory + a c7 script note.
pub fn compose_agent_body(staged: &Path, ctx: &BTreeMap<String, String>) -> String {
    let mut parts = vec![];
    let sp = strip_comments(&interp(&read_component(staged, "systemprompt.md"), ctx));
    if !sp.is_empty() {
        parts.push(sp);
    }
    let gr = strip_comments(&interp(&read_component(staged, "guardrails.md"), ctx));
    if !gr.is_empty() {
        parts.push(format!("# Operating Guardrails\n\n{gr}"));
    }
    let tg = strip_comments(&interp(&read_component(staged, "tool_descriptions/tool_guidance.md"), ctx));
    if !tg.is_empty() {
        parts.push(tg);
    }
    let mem = strip_comments(&interp(&read_component(staged, "LongTermMEMORY.md"), ctx));
    if !mem.is_empty() {
        parts.push(format!("# Long-Term Memory\n\n{mem}"));
    }
    let scripts = list_scripts(staged);
    if !scripts.is_empty() {
        let listing = scripts
            .iter()
            .map(|s| format!("- `python scripts/{}` — {}", s.file_name().unwrap().to_string_lossy(), script_doc(s)))
            .collect::<Vec<_>>()
            .join("\n");
        parts.push(format!("# Helper scripts (run via Bash)\n\nDeterministic helpers are available in `./scripts`. Prefer them over hand-rolling the same logic:\n\n{listing}"));
    }
    format!("{}\n", parts.into_iter().filter(|p| !p.trim().is_empty()).collect::<Vec<_>>().join("\n\n"))
}

fn ensure_mode_subagent(text: &str) -> String {
    if Regex::new(r"(?m)^mode:\s*subagent").unwrap().is_match(text) {
        return text.to_string();
    }
    if text.trim_start().starts_with("---") {
        return Regex::new(r"^---\n").unwrap().replacen(text, 1, "---\nmode: subagent\n").into_owned();
    }
    format!("---\nmode: subagent\ndescription: delegated subagent\n---\n\n{text}")
}

/// Materialise the staged tree into OpenCode-loadable artifacts under render_dir.
pub fn render_opencode_tree(staged: &Path, rd: &Path, ctx: &BTreeMap<String, String>) -> Result<()> {
    util::rmtree(rd);
    std::fs::create_dir_all(rd.join(".opencode/agent"))?;
    std::fs::create_dir_all(rd.join(".opencode/skills"))?;
    let body = compose_agent_body(staged, ctx);
    if !body.trim().is_empty() {
        util::write(&rd.join("AGENTS.md"), body)?;
    }
    let dirs = |sub: &str, file: &str| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(staged.join(sub))
            .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path().join(file)).filter(|p| p.is_file()).collect())
            .unwrap_or_default();
        v.sort();
        v
    };
    for skill in dirs("skills", "SKILL.md") {
        let d = skill.parent().unwrap();
        util::copy_tree(d, &rd.join(".opencode/skills").join(d.file_name().unwrap()))?;
    }
    for agent in dirs("sub_agents", "AGENT.md") {
        let name = agent.parent().unwrap().file_name().unwrap().to_string_lossy().to_string();
        util::write(&rd.join(".opencode/agent").join(format!("{name}.md")), ensure_mode_subagent(&util::read_lossy(&agent)?))?;
    }
    for s in list_scripts(staged) {
        util::write(&rd.join("scripts").join(s.file_name().unwrap()), std::fs::read(&s)?)?;
    }
    let mut plugins: Vec<PathBuf> = std::fs::read_dir(staged.join("plugin"))
        .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|e| e == "ts").unwrap_or(false)).collect())
        .unwrap_or_default();
    plugins.sort();
    for p in plugins {
        if p.file_name().unwrap().to_string_lossy().starts_with('_') {
            continue;
        }
        util::write(&rd.join(".opencode/plugin").join(p.file_name().unwrap()), std::fs::read(&p)?)?;
    }
    Ok(())
}

pub fn make_spec(substrate: &str) -> Result<BenchSpec> {
    let mode = match substrate {
        "oc" => "workspace_agents_md",
        "cx" => "codex_root",
        s => bail!("JobBench substrate must be 'oc' or 'cx', got {s:?}"),
    };
    let ch = resolve_channel_model(None, None, "high", 120, 3600)?;
    let mut spec = BenchSpec::new("jb", config::get().runs.join("jb"), mode, jobbench_prompts("", ""), channel_map(ch.clone(), ch));
    spec.domain_word = "profession".into();
    spec.leak_globs = ["RUBRICS.json", "eval_result/**", "expected/**", "reference/**"].iter().map(|s| s.to_string()).collect();
    spec.allow_subagents = substrate == "oc";
    spec.obs_trunc = config::JB_OBS_TRUNC;
    spec.checked()
}

fn extractor_lacks_deps(got: &Value) -> bool {
    got["files"].as_object().map(|f| f.values().any(|v| v.as_str().map(|s| s.starts_with("[ERROR:") && s.contains("not available")).unwrap_or(false))).unwrap_or(false)
}

fn has_events(p: &Path) -> bool {
    util::read_lossy(p).map(|t| t.lines().any(|l| l.trim_start().starts_with('{'))).unwrap_or(false)
}

/// True when the JUDGE failed, as opposed to the run merely scoring badly.
fn judge_failed(got: &Value) -> bool {
    let rubrics = got.get("rubrics").or_else(|| got.get("details")).and_then(|r| r.as_array()).cloned().unwrap_or_default();
    if let Some(e) = got.get("error").and_then(|e| e.as_str()) {
        if !e.trim().is_empty() && rubrics.is_empty() {
            return true;
        }
    }
    for k in ["overall_reasoning", "reasoning", "error"] {
        if got.get(k).and_then(|v| v.as_str()).map(|v| v.to_lowercase().contains("judge failed")).unwrap_or(false) {
            return true;
        }
    }
    rubrics.iter().filter(|r| r.is_object()).any(|r| {
        let v = r["result"]["overall_reasoning"].as_str().or_else(|| r["overall_reasoning"].as_str()).unwrap_or("").to_lowercase();
        JUDGE_FAILURE_PREFIXES.iter().any(|p| v.starts_with(p))
    })
}

/// Weighted total_score / max_score, with a labelled unweighted fallback.
fn parse_details(got: &Value) -> Option<Value> {
    let results: Vec<Value> = got["rubrics"].as_array().cloned().unwrap_or_default().iter().map(|r| r.get("result").cloned().unwrap_or(json!({}))).collect();
    let n_c: i64 = results.iter().map(|r| r["criteria_count"].as_i64().unwrap_or(0)).sum();
    let n_p: i64 = results.iter().map(|r| r["criteria_passed"].as_i64().unwrap_or(0)).sum();
    let total = got["total_score"].as_f64().unwrap_or(0.0);
    let mx = got["max_score"].as_f64().unwrap_or(0.0);
    if mx == 0.0 {
        if n_c <= 0 {
            return None;
        }
        return Some(json!({"n_passed": n_p, "n_criteria": n_c, "all_pass": n_p == n_c, "metric": "unweighted-fallback"}));
    }
    Some(json!({
        "n_passed": total, "n_criteria": mx, "all_pass": n_c > 0 && n_p == n_c, "metric": "weighted",
        "criteria_passed": n_p, "criteria_count": n_c, "judge_model": got["judge_model"].as_str().unwrap_or(""),
    }))
}

const EXTRACT_DRIVER: &str = "import importlib.util, json, sys\nfrom pathlib import Path\njudge_py, run_dir = sys.argv[1], Path(sys.argv[2])\nspec = importlib.util.spec_from_file_location(\"_jb_judge_extractor\", judge_py)\nmod = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(mod)\nfiles = {}\nfor f in sorted(p for p in run_dir.rglob(\"*\") if p.is_file()):\n    rel = str(f.relative_to(run_dir))\n    try:\n        files[rel] = str(mod.convert_file_to_text(f))\n    except Exception as exc:\n        files[rel] = f\"(extraction failed: {type(exc).__name__}: {exc})\"\nprint(json.dumps({\n    \"files\": files,\n    \"flat\": str(mod.extract_all_file_contents(run_dir)),\n    \"images\": [str(p) for p in mod.collect_image_paths(run_dir)],\n}))\n";

pub struct JobBenchAdapter {
    spec: BenchSpec,
    substrate: String,
    split_file: Option<PathBuf>,
    timeout: u64,
    store: PathBuf,
}

impl JobBenchAdapter {
    pub fn new(substrate: &str, split_file: Option<&Path>) -> Result<JobBenchAdapter> {
        Ok(JobBenchAdapter {
            spec: make_spec(substrate)?,
            substrate: substrate.into(),
            split_file: split_file.map(|p| p.to_path_buf()),
            timeout: if substrate == "oc" { 2400 } else { 3600 },
            store: config::get().runs.join("jb/dataset"),
        })
    }

    fn dataset_task(&self, task_id: &str) -> Result<PathBuf> {
        if !task_id.contains('/') || task_id.contains("..") || task_id.starts_with('/') {
            bail!("malformed JobBench task id {task_id:?} (expected '<profession>/task<N>')");
        }
        Ok(config::get().jb_repo().join("dataset/main").join(task_id))
    }
    fn out_dir(&self, task_id: &str, label: &str) -> PathBuf {
        self.store.join(task_id).join("model_output").join(label)
    }
    fn traj_dir(&self, task_id: &str, label: &str) -> PathBuf {
        self.store.join(task_id).join("model_traj").join(label)
    }
    /// `<store>/<profession>/<taskN>/model_output/<label>` -> (task_id, label).
    fn task_and_label(&self, run_dir: &Path) -> Result<(String, String)> {
        let parts: Vec<String> = run_dir.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
        let Some(i) = parts.iter().position(|p| p == "model_output") else {
            bail!("{} is not a JobBench model_output run directory", run_dir.display());
        };
        if i < 2 {
            bail!("{} is not a JobBench model_output run directory", run_dir.display());
        }
        Ok((parts[i - 2..i].join("/"), run_dir.file_name().unwrap().to_string_lossy().to_string()))
    }

    /// Copy the clean task inputs from the dataset into the run store, once, atomically.
    fn ensure_store_task(&self, task_id: &str) -> Result<PathBuf> {
        let dst = self.store.join(task_id).join("task_folder");
        if dst.is_dir() {
            return Ok(dst);
        }
        let src = self.dataset_task(task_id)?.join("task_folder");
        if !src.is_dir() {
            bail!("dataset task inputs missing: {} — re-materialise dependencies/job-bench-eval (recipe in dependencies/PINNED.md)", src.display());
        }
        std::fs::create_dir_all(dst.parent().unwrap())?;
        let _g = STORE_SEED_LOCK.lock().unwrap();
        if dst.is_dir() {
            return Ok(dst);
        }
        let tmp = dst.parent().unwrap().join(format!(".task_folder.tmp.{}.{:?}", std::process::id(), std::thread::current().id()));
        util::copy_tree(&src, &tmp)?;
        if std::fs::rename(&tmp, &dst).is_err() {
            util::rmtree(&tmp);
            if !dst.is_dir() {
                bail!("could not seed {}", dst.display());
            }
        }
        Ok(dst)
    }

    fn output_fingerprint(&self, run_dir: &Path) -> Option<String> {
        self.delivered(run_dir).then(|| RM::fingerprint(Some(run_dir)))
    }

    fn materialise(&self, src: &Path) -> Result<Install> {
        if self.substrate == "oc" {
            OpencodeSubstrate.materialise(src, &self.spec.install_mode, &BTreeMap::new())
        } else {
            CodexSubstrate.materialise(src, &self.spec.install_mode, &BTreeMap::new())
        }
    }

    fn launch(&self, req: &RunRequest, install: &Install) -> Result<Vec<PathBuf>> {
        if req.run_ids.len() != 1 {
            bail!("JobBench rolls one run per invocation (no batch substrate here); got {} run ids", req.run_ids.len());
        }
        let (task_id, label, t) = (req.task_ref.clone(), req.run_ids[0].replace('/', "__"), req.timeout);
        let cfg = config::get();
        if self.substrate == "oc" {
            OpencodeSubstrate.preflight()?;
            let render_dir = tempfile::Builder::new().prefix(&format!("hrender_{label}_")).tempdir()?.keep();
            render_opencode_tree(&install.staging_dir, &render_dir, &render_ctx(&self.domain_of(&task_id)))?;
            let mut env = config::scrub_env(None, config::provider_keys(&cfg.solver_model_oc));
            for (k, v) in [
                ("TASK_IDS", task_id.clone()),
                ("LABEL", label.clone()),
                ("HARNESS_RENDER_DIR", render_dir.to_string_lossy().to_string()),
                ("AGENT_NAME", String::new()),
                ("TASKS_BASE_DIR", self.store.to_string_lossy().to_string()),
                ("OPENCODE_DIR", cfg.opencode_dir.to_string_lossy().to_string()),
                ("MODEL_ID", cfg.solver_model_oc.clone()),
                ("TIMEOUT_PER_TASK", t.to_string()),
                ("MAX_RETRIES", "2".into()),
                ("MAX_CONCURRENT", "1".into()),
                ("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", OUTPUT_TOKEN_MAX.to_string()),
            ] {
                env.insert(k.into(), v);
            }
            let exe = std::env::current_exe()?;
            let res = self.run_runner(&[exe.to_string_lossy().to_string(), "jb-oc-runner".into()], env, 2 * t + 600, &label);
            util::rmtree(&render_dir);
            res?;
        } else {
            CodexSubstrate.preflight(false)?;
            if let Some(stale) = codex_config_error() {
                bail!(stale);
            }
            let mut env = CodexSubstrate.exec_env(&cfg.codex_home_dir);
            for (k, v) in [
                ("HARNESS2_CODEX_BIN", cfg.codex_dist.join("codex").to_string_lossy().to_string()),
                ("TASK_IDS", task_id.clone()),
                ("LABEL", label.clone()),
                ("HARNESS_RENDER_DIR", install.staging_dir.to_string_lossy().to_string()),
                ("TASKS_BASE_DIR", self.store.to_string_lossy().to_string()),
                ("BENCHMARK_MODELS", cfg.solver_model_cx.clone()),
                ("MAX_CONCURRENT_PER_MODEL", "1".into()),
                ("TIMEOUT_PER_TASK", t.to_string()),
                ("VERTEX_PROXY_KEY", cfg.vertex_proxy_key.clone()),
            ] {
                env.insert(k.into(), v);
            }
            let runner = runner_dir().join("run_benchmark_codex_cli_adapt.sh");
            if !runner.is_file() {
                bail!("JobBench runner missing: {}; restore third_party/jb_runner/.", runner.display());
            }
            self.run_runner(&["bash".into(), runner.to_string_lossy().to_string()], env, t + 600, &label)?;
        }
        Ok(vec![self.require_artifacts(&task_id, &label)?])
    }

    /// Run the launcher; timeouts leave the rollout incomplete.
    fn run_runner(&self, argv: &[String], mut env: BTreeMap<String, String>, wall: u64, label: &str) -> Result<()> {
        let cfg = config::get();
        let path = env.get("PATH").cloned().unwrap_or_default();
        env.insert(
            "PATH".into(),
            format!("{}:{}:{path}", cfg.jb_judge_python().parent().unwrap().display(), cfg.bun.parent().unwrap().display()),
        );
        match run_detached(argv, &env, None, wall, None)? {
            Some(0) => Ok(()),
            Some(code) => bail!("Benchmark runner exited with status {code}"),
            None => bail!("runner wall-timeout ({wall}s) for {label}; artifacts left UNMARKED so the next attempt re-rolls instead of adopting a truncated rollout"),
        }
    }

    fn require_artifacts(&self, task_id: &str, label: &str) -> Result<PathBuf> {
        let (out, traj) = (self.out_dir(task_id, label), self.traj_dir(task_id, label));
        let has_out = out.is_dir() && std::fs::read_dir(&out).map(|mut d| d.next().is_some()).unwrap_or(false);
        if !has_out {
            let has_traj = traj.is_dir() && jsonl(&traj).iter().any(|p| has_events(p));
            bail!(
                "JobBench produced no output for {task_id} @ {label} ({}: proxy/auth/binary) -- left UNMARKED so the next attempt re-rolls instead of serving an empty rollout as evidence",
                if has_traj { "trajectory without deliverables" } else { "dead agent" }
            );
        }
        Ok(out)
    }

    /// The judge's own extractor in a subprocess; raw text reads as the visible fallback.
    fn extract(&self, run_dir: &Path) -> Value {
        let key = std::fs::canonicalize(run_dir).unwrap_or_else(|_| run_dir.to_path_buf()).to_string_lossy().to_string();
        if let Some(h) = EXTRACT_CACHE.lock().unwrap().get(&key) {
            return h.clone();
        }
        let cfg = config::get();
        let judge_py = cfg.jb_repo().join("eval/judge.py");
        let py = if cfg.jb_judge_python().is_file() { cfg.jb_judge_python() } else { util::which("python3").unwrap_or_else(|| "python3".into()) };
        let mut got: Option<Value> = None;
        if judge_py.is_file() {
            let mut c = std::process::Command::new(py);
            c.arg("-c").arg(EXTRACT_DRIVER).arg(&judge_py).arg(&key);
            config::apply_env(&mut c, &config::scrub_env(None, &[]));
            if let Ok(o) = util::run_with_timeout(c, None, Duration::from_secs(900)) {
                if o.code == Some(0) {
                    got = serde_json::from_str::<Value>(&o.stdout).ok().filter(|g| g.is_object() && !extractor_lacks_deps(g));
                }
            }
        }
        let got = match got.filter(|g| g.get("files").is_some()) {
            Some(g) => g,
            None => {
                say!("  [jb] judge extractor unavailable for {key}; serving raw text reads");
                raw_extract(run_dir)
            }
        };
        EXTRACT_CACHE.lock().unwrap().insert(key, got.clone());
        got
    }

    /// (reporting judge, configured judge) detail files, in serving order.
    fn details_candidates(&self, task_id: &str, label: &str) -> (Option<PathBuf>, PathBuf) {
        let er = self.store.join(task_id).join("eval_result");
        let jm = config::get().jb_judge_model();
        let safe = jm.rsplit('/').next().unwrap_or("").replace('.', "-");
        (None, er.join(format!("eval_{label}")).join(format!("{safe}_judge.json")))
    }

    fn read_details(&self, details: &Path) -> Option<Value> {
        let got: Value = serde_json::from_str(&std::fs::read_to_string(details).ok()?).ok()?;
        if !got.is_object() {
            return None;
        }
        if judge_failed(&got) {
            let q = details.with_file_name(format!("{}.judge_failed", details.file_name().unwrap().to_string_lossy()));
            if let Err(e) = std::fs::rename(details, &q) {
                say!("  [jb] could not quarantine judge-failed details {}: {e}", details.display());
            }
            return None;
        }
        if let (Some(r), Some(tc)) = (got["rubrics"].as_array(), got["total_count"].as_i64()) {
            if tc > 0 && (r.len() as i64) < tc {
                say!("  [jb] partial judge details ({}/{tc} rubrics) at {}; not served, judge will resume", r.len(), details.display());
                return None;
            }
        }
        Some(got)
    }

    /// A run that delivered nothing is a real zero, built from the rubric weights.
    fn no_output_score(&self, task_id: &str) -> Option<Value> {
        let rp = self.dataset_task(task_id).ok()?.join("RUBRICS.json");
        let v: Value = serde_json::from_str(&std::fs::read_to_string(rp).ok()?).ok()?;
        let (mut mx, mut n_c) = (0.0, 0usize);
        for r in v["rubrics"].as_array().cloned().unwrap_or_default() {
            if !r.is_object() {
                continue;
            }
            mx += r["weight"].as_f64().unwrap_or(0.0);
            n_c += match &r["criterion"] {
                Value::String(_) => 1,
                Value::Array(a) => a.len(),
                _ => 0,
            };
        }
        if mx == 0.0 {
            return None;
        }
        Some(json!({"n_passed": 0.0, "n_criteria": mx, "all_pass": false, "metric": "weighted", "criteria_passed": 0, "criteria_count": n_c, "no_output": true}))
    }

    /// Invoke the upstream judge CLI in a subprocess; false on any failure.
    fn run_judge(&self, task_id: &str, label: &str, run_dir: &Path) -> bool {
        let cfg = config::get();
        let judge_py = cfg.jb_repo().join("eval/judge.py");
        let Ok(rubrics) = self.dataset_task(task_id).map(|d| d.join("RUBRICS.json")) else { return false };
        for (need, what) in [(&judge_py, "upstream judge"), (&rubrics, "task rubrics")] {
            if !need.is_file() {
                say!("  [jb] {what} missing ({}); judge not run", need.display());
                return false;
            }
        }
        let py = cfg.jb_judge_python();
        if !py.is_file() {
            say!("  [jb] judge interpreter missing ({}); run `harness2 setup jb` -- judge not run (None, never 0)", py.display());
            return false;
        }
        let token = if !cfg.jb_judge_api_key.is_empty() { cfg.jb_judge_api_key.clone() } else { gcloud_token() };
        if token.is_empty() {
            say!("  [jb] no Vertex access token (gcloud auth print-access-token); judge not run");
            return false;
        }
        let (_r, own) = self.details_candidates(task_id, label);
        let _ = std::fs::create_dir_all(own.parent().unwrap());
        let mut c = std::process::Command::new(&py);
        c.arg(&judge_py)
            .arg("--output-dir").arg(run_dir)
            .arg("--rubrics-file").arg(&rubrics)
            .arg("--details-file").arg(&own)
            .arg("--judge-model").arg(cfg.jb_judge_model())
            .arg("--api-base").arg(&cfg.jb_judge_api_base)
            .args(["--max-workers", "8", "--max-retries", "3", "--timeout-per-rubric", "240", "--evaluated-model", label]);
        let mut env = config::scrub_env(None, &[]);
        env.insert("JUDGE_API_KEY".into(), token);
        env.insert("JUDGE_REASONING_EFFORT".into(), config::JB_JUDGE_REASONING_EFFORT.into());
        env.insert("VERTEXAI_LOCATION".into(), cfg.vertex_region.clone());
        config::apply_env(&mut c, &env);
        match util::run_with_timeout(c, None, Duration::from_secs(1800)) {
            Ok(o) if o.code == Some(0) => true,
            Ok(o) => {
                say!("  [jb] judge exited {:?} for {label}: {}", o.code, util::tail_chars(&o.stderr, 300));
                false
            }
            Err(e) => {
                say!("  [jb] judge subprocess failed for {label}: {e}");
                false
            }
        }
    }
}

pub fn gcloud_token() -> String {
    let mut c = std::process::Command::new("gcloud");
    c.args(["auth", "application-default", "print-access-token"]);
    util::run_with_timeout(c, None, Duration::from_secs(60)).map(|o| o.stdout.trim().to_string()).unwrap_or_default()
}

fn jsonl(d: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(d)
        .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|e| e == "jsonl").unwrap_or(false)).collect())
        .unwrap_or_default()
}

fn raw_extract(d: &Path) -> Value {
    let (mut files, mut parts, mut images) = (serde_json::Map::new(), vec![], vec![]);
    for f in util::files_sorted(d) {
        let rel = util::rel(d, &f);
        let ext = f.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if VISION_EXTS.contains(&ext.as_str()) {
            images.push(f.to_string_lossy().to_string());
        }
        let text = util::read_lossy(&f).unwrap_or_else(|e| format!("(unreadable: {e})"));
        let clipped = if util::char_len(&text) <= MAX_CHARS_PER_FILE {
            text.clone()
        } else {
            format!("{}\n... [Content truncated at {MAX_CHARS_PER_FILE} characters]", util::take_chars(&text, MAX_CHARS_PER_FILE))
        };
        parts.push(format!("=== FILE: {} ===\n{clipped}\n", f.file_name().unwrap().to_string_lossy()));
        files.insert(rel, json!(text));
    }
    images.truncate(MAX_VISION);
    json!({"files": files, "flat": parts.join("\n"), "images": images})
}

impl BenchmarkAdapter for JobBenchAdapter {
    fn spec(&self) -> &BenchSpec {
        &self.spec
    }
    fn substrate_name(&self) -> &str {
        &self.substrate
    }
    fn surface(&self) -> SurfaceSpec {
        if self.substrate == "oc" { OpencodeSubstrate.surface() } else { CodexSubstrate.surface() }
    }
    fn tasks(&self, split: &str) -> Result<Vec<String>> {
        tasks::select(&tasks::discover("jb", None)?, split, self.split_file.as_deref(), "jb")
    }
    fn domain_of(&self, task_id: &str) -> String {
        task_id.split('/').next().unwrap_or("").to_string()
    }
    fn reference_domain_order(&self) -> Option<Vec<String>> {
        let mut v: Vec<String> = tasks::discover("jb", None).ok()?.into_values().collect();
        v.sort();
        v.dedup();
        Some(v)
    }
    fn prompts_for(&self, task_id: &str) -> BenchPrompts {
        let d = self.domain_of(task_id);
        jobbench_prompts(&d, &d)
    }
    fn task_prompt(&self, task_id: &str) -> Result<String> {
        let p = self.dataset_task(task_id)?.join("task_folder/TASK_INSTRUCTIONS.txt");
        if !p.is_file() {
            bail!("task instructions missing: {} — re-materialise dependencies/job-bench-eval's dataset/main (recipe in dependencies/PINNED.md)", p.display());
        }
        util::read_lossy(&p)
    }
    fn task_files(&self, task_id: &str) -> Option<PathBuf> {
        let d = self.dataset_task(task_id).ok()?.join("task_folder");
        d.is_dir().then_some(d)
    }
    /// One rollout; the label carries the whole namespace (run_id, / rewritten __).
    fn execute(&self, task_id: &str, harness_dir: Option<&Path>, run_id: &str, timeout: Option<u64>) -> Result<PathBuf> {
        let label = run_id.replace('/', "__");
        let (out, mark) = (self.out_dir(task_id, &label), self.traj_dir(task_id, &label));
        let inputs = RM::inputs_fingerprint(&self.task_prompt(task_id)?, self.task_files(task_id).as_deref());
        if RM::is_complete(&mark, harness_dir, true, Some(&inputs)) && self.artifacts_intact(&out) == Some(true) {
            std::fs::create_dir_all(&out)?;
            return Ok(out);
        }
        if out.is_dir() || mark.is_dir() {
            let mut reason = RM::why_not(&mark, harness_dir, Some(&inputs));
            if reason == "reusable" {
                reason = "output fingerprint missing or changed".into();
            }
            say!("  [jb-{}] re-rolling {label}: {reason}", self.substrate);
        }
        util::rmtree(&out);
        util::rmtree(&mark);
        EXTRACT_CACHE.lock().unwrap().retain(|k, _| !k.ends_with(&label));
        let (_reporting, own) = self.details_candidates(task_id, &label);
        if own.parent().map(|p| p.is_dir()).unwrap_or(false) {
            util::rmtree(own.parent().unwrap());
        }
        self.ensure_store_task(task_id)?;
        let base = self.base_harness()?;
        let install = self.materialise(harness_dir.unwrap_or(&base))?;
        let res = self.launch(&RunRequest { task_ref: task_id.into(), run_ids: vec![run_id.into()], timeout: timeout.unwrap_or(self.timeout) }, &install);
        install.drop_staging();
        let dirs = res?;
        std::fs::create_dir_all(&out)?;
        let mut extra = serde_json::Map::new();
        extra.insert("output_fingerprint".into(), json!(self.output_fingerprint(&out)));
        RM::mark_complete(&mark, run_id, harness_dir, Some(&inputs), Some(extra))?;
        Ok(dirs[0].clone())
    }
    fn artifacts_intact(&self, run_dir: &Path) -> Option<bool> {
        let (task_id, label) = self.task_and_label(run_dir).ok()?;
        let td = self.traj_dir(&task_id, &label);
        let marker = RM::read_mark(&td).unwrap_or(json!({}));
        Some(
            marker.get("output_fingerprint").is_some()
                && marker["output_fingerprint"] == json!(self.output_fingerprint(run_dir))
                && (self.delivered(run_dir) || jsonl(&td).iter().any(|p| has_events(p))),
        )
    }
    fn delivered(&self, run_dir: &Path) -> bool {
        run_dir.is_dir() && !util::files_sorted(run_dir).is_empty()
    }
    /// Newest attempt's event stream, via the substrate's renderer.
    fn trajectory(&self, run_dir: &Path, obs_trunc: usize) -> Result<String> {
        let (task_id, label) = self.task_and_label(run_dir)?;
        let td = self.traj_dir(&task_id, &label);
        let mut files = jsonl(&td);
        files.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).and_then(|m| m.modified()).ok()));
        let Some(f) = files.first() else { return Ok(format!("(no trajectory for {task_id} @ {label})")) };
        Ok(if self.substrate == "oc" { OpencodeSubstrate.read_trajectory(f, obs_trunc, 8_000_000) } else { CodexSubstrate.read_trajectory(f, obs_trunc, 8_000_000) })
    }
    fn deliverable_files(&self, run_dir: &Path) -> Option<BTreeMap<String, String>> {
        if !run_dir.is_dir() {
            return Some(BTreeMap::new());
        }
        Some(self.extract(run_dir)["files"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default())
    }
    fn deliverable_images(&self, run_dir: &Path) -> Vec<PathBuf> {
        if !run_dir.is_dir() {
            return vec![];
        }
        self.extract(run_dir)["images"].as_array().cloned().unwrap_or_default().iter().filter_map(|v| v.as_str().map(PathBuf::from)).collect()
    }
    fn deliverable_text(&self, run_dir: &Path) -> Result<String> {
        if !run_dir.is_dir() {
            return Ok(String::new());
        }
        Ok(self.extract(run_dir)["flat"].as_str().unwrap_or("").to_string())
    }
    fn base_harness(&self) -> Result<PathBuf> {
        let base = runner_dir().join(if self.substrate == "oc" { "harness_baseline_oc" } else { "codex_base_harness" });
        if !tree::is_tree(&self.surface(), &base) {
            bail!(
                "JobBench base harness missing or not a {} tree: {} — restore third_party/jb_runner/; an absent base would make every base rollout silently harness-free",
                self.surface().name,
                base.display()
            );
        }
        Ok(base)
    }
    fn validate_tree(&self, _t: &Tree) -> Result<()> {
        Ok(())
    }
    fn preflight(&self) -> Result<()> {
        let cfg = config::get();
        if self.substrate == "oc" {
            OpencodeSubstrate.preflight()?;
        } else {
            CodexSubstrate.preflight(false)?;
        }
        let judge_py = cfg.jb_repo().join("eval/judge.py");
        if !judge_py.is_file() {
            bail!("upstream JobBench judge missing at {}; run `harness2 setup jb`", judge_py.display());
        }
        if self.substrate == "cx" {
            config::require("HARNESS2_CODEX_API_KEY", "JB codex cells")?;
            if let Some(s) = codex_config_error() {
                bail!(s);
            }
        }
        let py = cfg.jb_judge_python();
        let remedy = "harness2 setup jb";
        if !py.is_file() {
            bail!("JB judge interpreter missing at {}; {remedy}", py.display());
        }
        let mut c = std::process::Command::new(&py);
        c.args(["-c", "import pandas, openpyxl, docx"]);
        let o = util::run_with_timeout(c, None, Duration::from_secs(300)).map_err(|e| anyhow::anyhow!("JB judge interpreter {} could not be run: {e}; {remedy}", py.display()))?;
        if o.code != Some(0) {
            bail!(
                "JB judge interpreter {} lacks the extraction deps (pandas, openpyxl, python-docx): {} -- {remedy}. Without them the judge grades '[ERROR: ... not available]' text as a real zero.",
                py.display(),
                util::tail_chars(o.stderr.trim(), 300)
            );
        }
        if cfg.jb_judge_api_base.is_empty() {
            say!("  [jb] WARNING: HARNESS2_JUDGE_BASE_URL is empty; rollouts will run but judging refuses until it is set (see .env.example)");
        }
        Ok(())
    }
    /// Judge one run. None means the judge FAILED, which is not a zero.
    fn score(&self, run_dir: &Path, task_id: &str) -> Result<Option<Value>> {
        let label = run_dir.file_name().unwrap().to_string_lossy().to_string();
        let (reporting, own) = self.details_candidates(task_id, &label);
        for d in reporting.iter().chain(std::iter::once(&own)) {
            if let Some(got) = self.read_details(d) {
                return Ok(parse_details(&got));
            }
        }
        if !self.delivered(run_dir) {
            return Ok(self.no_output_score(task_id));
        }
        if !self.run_judge(task_id, &label, run_dir) {
            return Ok(None);
        }
        Ok(self.read_details(&own).and_then(|g| parse_details(&g)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompts_and_lookup() {
        let bp = jobbench_prompts("Management", "lawyers");
        assert!(bp.domain_guidance.contains("DOMAIN LENS — MANAGEMENT"));
        assert!(bp.domain_guidance.contains("PROFESSION LENS — Fix jurisdiction"));
        assert_eq!(bp.prompt_pack, "tts");
        bp.checked().unwrap();
    }
    #[test]
    fn details_parsing() {
        let v = json!({"rubrics": [{"result": {"criteria_count": 4, "criteria_passed": 4}}], "total_score": 7.0, "max_score": 10.0});
        let p = parse_details(&v).unwrap();
        assert_eq!(p["n_passed"], 7.0);
        assert_eq!(p["all_pass"], true);
        assert!(judge_failed(&json!({"error": "boom"})));
    }
    #[test]
    fn body_composition() {
        let d = tempfile::tempdir().unwrap();
        util::write(&d.path().join("systemprompt.md"), "Hello {{ domain_focus }} <!-- c -->").unwrap();
        util::write(&d.path().join("scripts/x.py"), "\"\"\"Sum things.\"\"\"\n").unwrap();
        let mut ctx = BTreeMap::new();
        ctx.insert("domain_focus".to_string(), "law".to_string());
        let b = compose_agent_body(d.path(), &ctx);
        assert!(b.starts_with("Hello law"));
        assert!(b.contains("`python scripts/x.py` — Sum things."));
    }
}
