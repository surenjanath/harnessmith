//! `setup` (install a benchmark's pinned tools, runner overlays, dependencies and data),
//! `check` (readiness), `bridge` (Vertex Responses endpoint for Codex), and the shared
//! preflight that `run` calls before spending anything.

use super::common::{self, CellArgs};
use crate::config::{self, ConfigError};
use crate::core::spec::BenchmarkAdapter;
use crate::substrate::opencode::OpencodeSubstrate;
use crate::util;
use anyhow::{anyhow, bail, Result};
use clap::Args;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub const PATCHES: &[(&str, &str)] = &[
    ("opencode", "opencode-compat.diff"),
    ("job-bench-eval", "job-bench-eval-judge.diff"),
    ("harvey-labs", "harvey-labs-judge-vertex-anthropic.diff"),
    ("workbuddy-bench", "workbuddy-bench-runtime.diff"),
];
pub const RUNNERS: &[&str] = &["jb_runner/run_benchmark_codex_cli_adapt.sh", "lab_runner/opencode/run_opencode.py", "wb_runner/agents/oc_agent.py"];

fn patch_of(name: &str) -> &'static str {
    PATCHES.iter().find(|(n, _)| *n == name).map(|(_, p)| *p).unwrap_or("")
}

pub fn repo_name(bench: &str) -> Result<&'static str> {
    Ok(match bench {
        "jb" => "job-bench-eval",
        "lab" => "harvey-labs",
        "wb" => "workbuddy-bench",
        b => bail!("unknown benchmark {b:?}"),
    })
}

/// Which setup runners or patches are absent from third_party/, if any.
pub fn missing_runners() -> String {
    let tp = &config::get().third_party;
    let need: Vec<PathBuf> = PATCHES.iter().map(|(_, p)| tp.join("patches").join(p)).chain(RUNNERS.iter().map(|r| tp.join(r))).collect();
    let absent = need.iter().filter(|p| !p.exists()).count();
    if absent == 0 {
        return String::new();
    }
    format!("benchmark runners and patches are not included in this copy ({absent} of {} missing under third_party/); see third_party/README.md", need.len())
}

fn show(cmd: &[String], cwd: Option<&Path>) -> String {
    let joined = cmd.iter().map(|c| if c.contains(' ') { format!("'{c}'") } else { c.clone() }).collect::<Vec<_>>().join(" ");
    match cwd {
        Some(d) => format!("[{}] {joined}", d.display()),
        None => joined,
    }
}

/// Print and (unless dry) run a command, failing on a nonzero exit.
pub fn run(cmd: &[String], cwd: Option<&Path>, dry: bool) -> Result<()> {
    println!("{}", show(cmd, cwd));
    if dry {
        return Ok(());
    }
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]);
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    let st = c.status().map_err(|e| anyhow!("{}: {e}", cmd[0]))?;
    if !st.success() {
        bail!("command failed ({st}): {}", show(cmd, cwd));
    }
    Ok(())
}

fn s<T: AsRef<std::ffi::OsStr>>(x: T) -> String {
    x.as_ref().to_string_lossy().to_string()
}

fn git_out(args: &[&str]) -> Option<String> {
    let o = Command::new("git").args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

pub fn clone(name: &str, dry: bool) -> Result<()> {
    let repo = config::get().dependencies.join(name);
    let pin = config::pin(name);
    let r = s(&repo);
    if repo.join(".git").exists() {
        if let Some(head) = git_out(&["-C", &r, "rev-parse", "--verify", "HEAD"]) {
            if head == pin {
                return Ok(());
            }
            bail!("{} is at {head}, expected {pin}; move this checkout aside before setup", repo.display());
        }
        let origin = git_out(&["-C", &r, "remote", "get-url", "origin"]).unwrap_or_default();
        let other = std::fs::read_dir(&repo)?.filter_map(|e| e.ok()).any(|e| e.file_name() != ".git");
        if other || origin != config::origin(name) {
            bail!("Incomplete checkout at {}; move it aside before setup", repo.display());
        }
    } else {
        if repo.exists() && std::fs::read_dir(&repo)?.next().is_some() {
            bail!("Refusing to overwrite nonempty {}", repo.display());
        }
        run(&[s("git"), s("init"), s("-q"), r.clone()], None, dry)?;
        run(&[s("git"), s("-C"), r.clone(), s("remote"), s("add"), s("origin"), s(config::origin(name))], None, dry)?;
    }
    run(&[s("git"), s("-C"), r.clone(), s("fetch"), s("--depth"), s("1"), s("origin"), s(pin)], None, dry)?;
    run(&[s("git"), s("-C"), r, s("checkout"), s("--detach"), s("FETCH_HEAD")], None, dry)
}

pub fn patch(name: &str, dry: bool) -> Result<()> {
    let repo = s(config::get().dependencies.join(name));
    let src = s(config::get().third_party.join("patches").join(patch_of(name)));
    if dry {
        run(&[s("git"), s("-C"), repo.clone(), s("apply"), s("--check"), src.clone()], None, true)?;
        return run(&[s("git"), s("-C"), repo, s("apply"), src], None, true);
    }
    let applied = Command::new("git").args(["-C", &repo, "apply", "--reverse", "--check", &src]).output().map(|o| o.status.success()).unwrap_or(false);
    if applied {
        return Ok(());
    }
    run(&[s("git"), s("-C"), repo.clone(), s("apply"), s("--check"), src.clone()], None, false)?;
    run(&[s("git"), s("-C"), repo, s("apply"), src], None, false)
}

pub fn overlays(name: &str, dry: bool) -> Result<()> {
    let tp = &config::get().third_party;
    let repo = config::get().dependencies.join(name);
    let pairs: Vec<(PathBuf, PathBuf)> = match name {
        "harvey-labs" => vec![
            (tp.join("lab_runner/opencode"), repo.join("opencode")),
            (tp.join("lab_runner/tts_lab"), repo.join("external/tts_lab")),
            (tp.join("lab_runner/adapt_runs_cx"), repo.join("external/adapt_runs_cx")),
        ],
        "workbuddy-bench" => vec![
            (tp.join("wb_runner/agents"), repo.join("src/workbuddy_bench/agents")),
            (tp.join("wb_runner/configs/harnesses"), repo.join("configs/harnesses")),
        ],
        _ => vec![],
    };
    for (src, dst) in pairs {
        println!("Copy runner assets: {} -> {}", src.display(), dst.display());
        if !dry {
            util::copy_tree_filtered(&src, &dst, |r| !r.contains("__pycache__") && !r.ends_with(".pyc") && !r.ends_with("PROVENANCE.md"))?;
        }
    }
    Ok(())
}

/// Generate the two Vertex routes from the user's model IDs, never credentials.
pub fn workbuddy_models(dry: bool) -> Result<()> {
    let cfg = config::get();
    let solver = cfg.solver_model_oc.trim_start_matches("google-vertex/").to_string();
    let judge = cfg.judge_model.clone();
    let mut problem = "";
    if !cfg.solver_model_oc.starts_with("google-vertex/gemini") || !judge.starts_with("gemini") {
        problem = "WorkBuddy setup requires a google-vertex/gemini... executor and a gemini... judge; choose IDs available in your project";
    } else if cfg.vertex_project.is_empty() {
        problem = "Set GOOGLE_CLOUD_PROJECT for WorkBuddy's Vertex routes";
    }
    if !problem.is_empty() {
        if !dry {
            return Err(anyhow!(ConfigError(problem.into())));
        }
        println!("Configure WorkBuddy solver and judge routes -- {problem}");
        return Ok(());
    }
    for (role, model) in [("solver", &solver), ("judge", &judge)] {
        let payload = json!({"model": {
            "name": format!("google/{model}"), "protocols": ["openai"], "backend_url_env": "HARNESS2_WB_VERTEX_URL",
            "backend_key_command": "gcloud auth application-default print-access-token", "backend_key_ttl": 1800,
            "thought_signature_passback": true,
            "params": {"max_output_tokens": if role == "solver" { 64000 } else { 16384 }, "extra_body": {"reasoning_effort": if role == "solver" { "high" } else { "low" }}},
        }});
        let p = cfg.wb_repo().join("configs/models/harness2").join(format!("{role}.yaml"));
        println!("Configure WorkBuddy {role}: {model}");
        if !dry {
            util::write(&p, format!("{}\n", serde_json::to_string_pretty(&payload)?))?;
        }
    }
    Ok(())
}

pub const CODEX_CONFIG_REMEDY: &str = "rerun `harness2 setup <bench> --substrate cx`";
pub const CODEX_PROVIDER: &str = "harness2";

pub fn configure_codex(dry: bool) -> Result<()> {
    let cfg = config::get();
    config::require("HARNESS2_CODEX_BASE_URL", "Codex setup")?;
    let body = format!(
        "model = {}\nmodel_provider = {}\n[model_providers.{CODEX_PROVIDER}]\nname = \"Configured Responses endpoint\"\nbase_url = {}\nwire_api = \"responses\"\nenv_key = \"VERTEX_PROXY_KEY\"\n",
        json!(cfg.solver_model_cx),
        json!(CODEX_PROVIDER),
        json!(cfg.codex_base_url)
    );
    if !dry {
        util::write(&cfg.codex_home_dir.join("config.toml"), body)?;
    }
    Ok(())
}

/// Why `CODEX_HOME/config.toml` no longer matches the live Codex configuration, if it does not.
pub fn codex_config_error() -> Option<String> {
    let cfg = config::get();
    let p = cfg.codex_home_dir.join("config.toml");
    let Ok(text) = std::fs::read_to_string(&p) else {
        return Some(format!(
            "CODEX_HOME has no readable config.toml: {} -- the configured provider block there routes Codex to your Responses endpoint; {CODEX_CONFIG_REMEDY}",
            p.display()
        ));
    };
    // The file is written by `configure_codex`: read back its `key = "json string"` lines.
    let get = |section: &str, key: &str| -> Option<String> {
        let mut cur = String::new();
        for line in text.lines() {
            let l = line.trim();
            if l.starts_with('[') {
                cur = l.trim_matches(['[', ']']).to_string();
                continue;
            }
            if cur == section {
                if let Some((k, v)) = l.split_once('=') {
                    if k.trim() == key {
                        return serde_json::from_str::<String>(v.trim()).ok();
                    }
                }
            }
        }
        None
    };
    let provider = get("", "model_provider").unwrap_or_default();
    let mut stale = vec![];
    let base = get(&format!("model_providers.{provider}"), "base_url");
    if base.as_deref() != Some(cfg.codex_base_url.as_str()) {
        stale.push(format!("base_url is {base:?}, configuration says {:?}", cfg.codex_base_url));
    }
    let model = get("", "model");
    if model.as_deref() != Some(cfg.solver_model_cx.as_str()) {
        stale.push(format!("model is {model:?}, configuration says {:?}", cfg.solver_model_cx));
    }
    (!stale.is_empty()).then(|| format!("{} is stale ({}); {CODEX_CONFIG_REMEDY}", p.display(), stale.join("; ")))
}

pub fn codex(dry: bool) -> Result<()> {
    let cfg = config::get();
    let root = cfg.dependencies.join("codex");
    run(&[s("npm"), s("install"), s("--prefix"), s(&root), format!("@openai/codex@{}", config::CODEX_VERSION)], None, dry)?;
    if !dry {
        let cands: Vec<PathBuf> = util::files_sorted(&root.join("node_modules/@openai"))
            .into_iter()
            .filter(|p| p.ends_with("bin/codex"))
            .collect();
        if cands.len() != 1 {
            bail!("Expected one native Codex binary in {}; found {}", root.display(), cands.len());
        }
        let platform = cands[0].parent().unwrap().parent().unwrap();
        util::copy_tree(platform, &cfg.codex_dist)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for p in util::files_sorted(&cfg.codex_dist) {
                if let Ok(m) = std::fs::metadata(&p) {
                    let mode = m.permissions().mode();
                    let extra = 0o044 | if mode & 0o100 != 0 { 0o011 } else { 0 };
                    let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode | extra));
                }
            }
            let bin = cfg.codex_dist.join("codex");
            let _ = std::fs::remove_file(&bin);
            std::os::unix::fs::symlink("bin/codex", &bin)?;
        }
    }
    configure_codex(dry)
}

/// Install the JobBench task tree from its public Hugging Face dataset repo.
pub fn download_jobbench(dry: bool) -> Result<()> {
    let repo = config::get().jb_repo();
    let dest = repo.join("dataset/main");
    if dest.join(".harness2-download.json").is_file() {
        return Ok(());
    }
    let code = "from huggingface_hub import HfApi, snapshot_download\nimport json, sys\nrevision = HfApi().dataset_info(\"JobBench/job-bench\").sha\nsnapshot_download(\"JobBench/job-bench\", repo_type=\"dataset\", revision=revision,\n                  allow_patterns=[\"dataset/**\"], local_dir=sys.argv[1])\nopen(sys.argv[1]+\"/revision.json\", \"w\").write(json.dumps({\"revision\": revision}))\n";
    if dry {
        println!("Download JobBench/job-bench dataset/** from Hugging Face; record its revision");
        return Ok(());
    }
    if dest.exists() && std::fs::read_dir(&dest)?.next().is_some() {
        bail!("{} already contains data without a download record; move it aside first", dest.display());
    }
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let tmp = tempfile::Builder::new().prefix("download-").tempdir_in(dest.parent().unwrap())?;
    run(&[s(repo.join(".venv/bin/python")), s("-c"), s(code), s(tmp.path())], None, false)?;
    let staged = tmp.path().join("dataset");
    let ok = std::fs::read_dir(&staged)?
        .filter_map(|e| e.ok())
        .any(|d| std::fs::read_dir(d.path()).map(|rd| rd.filter_map(|e| e.ok()).any(|t| t.path().join("task_folder/TASK_INSTRUCTIONS.txt").is_file())).unwrap_or(false));
    if !ok {
        bail!("JobBench download did not contain task instructions");
    }
    if dest.exists() {
        std::fs::remove_dir(&dest)?;
    }
    std::fs::rename(&staged, &dest)?;
    std::fs::copy(tmp.path().join("revision.json"), dest.join(".harness2-download.json"))?;
    Ok(())
}

#[derive(Args)]
pub struct SetupArgs {
    /// jb | lab | wb
    pub benchmark: String,
    /// wb only: one subset to install; omitted installs office, web, and code
    #[arg(long)]
    pub subset: Option<String>,
    #[arg(long, default_value = "oc")]
    pub substrate: String,
    #[arg(long)]
    pub dry_run: bool,
}

pub fn setup_cmd(a: SetupArgs) -> Result<i32> {
    if a.subset.is_some() && a.benchmark != "wb" {
        bail!("--subset only applies to wb");
    }
    let dry = a.dry_run;
    let missing = missing_runners();
    if !missing.is_empty() && !dry {
        bail!("setup failed: {missing}");
    }
    if !missing.is_empty() {
        println!("Note: {missing}");
    }
    let name = repo_name(&a.benchmark)?;
    let cfg = config::get();
    let res: Result<()> = (|| {
        if !dry {
            let mut need = vec!["git", "uv", "npm"];
            if a.benchmark == "wb" {
                need.extend(["docker", "curl"]);
            }
            for b in need {
                if util::which(b).is_none() {
                    bail!("Missing {b}; see README.md");
                }
            }
        }
        clone("opencode", dry)?;
        patch("opencode", dry)?;
        run(&[s("npm"), s("install"), s("--prefix"), s(cfg.dependencies.join("tools")), s("bun@1.3.11")], None, dry)?;
        run(&[s(&cfg.bun), s("install"), s("--frozen-lockfile"), s("--ignore-scripts")], Some(&cfg.opencode_dir), dry)?;
        clone(name, dry)?;
        patch(name, dry)?;
        overlays(name, dry)?;
        let repo = cfg.dependencies.join(name);
        run(&[s("uv"), s("sync"), s("--locked"), s("--python"), s("3.12")], Some(&repo), dry)?;
        match a.benchmark.as_str() {
            "jb" => {
                run(&[s("uv"), s("pip"), s("install"), s("--python"), s(repo.join(".venv/bin/python")), s("python-docx")], None, dry)?;
                download_jobbench(dry)?;
            }
            "lab" => run(&[s("uv"), s("pip"), s("install"), s("--python"), s(repo.join(".venv/bin/python")), s("google-auth[requests]")], None, dry)?,
            _ => {
                workbuddy_models(dry)?;
                let subsets: Vec<String> = a.subset.clone().map(|x| vec![x]).unwrap_or_else(|| vec!["office".into(), "web".into(), "code".into()]);
                let mut cmd = vec![s("bash"), s("scripts/dataset/fetch-dataset.sh")];
                cmd.extend(subsets);
                run(&cmd, Some(&repo), dry)?;
                if a.substrate == "oc" {
                    run(&[s("uv"), s("run"), s("--no-sync"), s("bash"), s("scripts/harness/build-harness-mounts.sh"), s("--harness"), s("opencode/1.14.18")], Some(&repo), dry)?;
                }
            }
        }
        if a.substrate == "cx" {
            codex(dry)?;
        }
        Ok(())
    })();
    if let Err(e) = res {
        bail!("setup failed: {e}");
    }
    println!("{}", if dry { "Setup plan complete." } else { "Installation complete. Run `harness2 check` before an experiment." });
    Ok(0)
}

fn probe(errors: &mut Vec<String>, label: &str, cmd: &[String], cwd: Option<&Path>, remedy: &str) -> bool {
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]);
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    match util::run_with_timeout(c, None, Duration::from_secs(45)) {
        Ok(o) if o.code == Some(0) => true,
        Ok(o) if o.timed_out => {
            errors.push(format!("{label}: TimeoutExpired; see README.md"));
            false
        }
        Ok(_) => {
            errors.push(format!("{label} failed; {remedy}"));
            false
        }
        Err(e) => {
            errors.push(format!("{label}: {e}; see README.md"));
            false
        }
    }
}

/// Benchmark dependency and runtime readiness checks.
pub fn check(adapter: &dyn BenchmarkAdapter, judge: bool) -> Vec<String> {
    let bench = adapter.spec().name.split('-').next().unwrap_or("").to_string();
    if bench == "goal" || bench == "demo" {
        let mut e = vec![];
        if let Err(x) = adapter.tasks("all") {
            e.push(x.to_string());
        }
        if let Err(x) = adapter.preflight() {
            e.push(x.to_string());
        }
        return e;
    }
    let missing = missing_runners();
    if !missing.is_empty() {
        return vec![missing];
    }
    let cfg = config::get();
    let mut errors = vec![];
    let remedy = "run `harness2 setup` for this benchmark";
    if let Err(e) = cfg.validate_models(judge) {
        errors.push(e.to_string());
    }
    if let Err(e) = adapter.tasks("all") {
        errors.push(e.to_string());
    }
    if let Err(e) = OpencodeSubstrate.preflight() {
        errors.push(e.to_string());
    }
    if let Err(e) = adapter.preflight() {
        errors.push(e.to_string());
    }
    let Ok(name) = repo_name(&bench) else { return errors };
    let repo = cfg.dependencies.join(name);
    for dep in [name, "opencode"] {
        let target = cfg.dependencies.join(dep);
        if target.join(".git").exists() {
            match git_out(&["-C", &s(&target), "rev-parse", "HEAD"]) {
                Some(h) if h != config::pin(dep) => errors.push(format!("{dep}: checkout differs from the supported pin")),
                None => errors.push(format!("{dep}: cannot read checkout revision")),
                _ => {}
            }
        }
        let p = cfg.third_party.join("patches").join(patch_of(dep));
        probe(&mut errors, &format!("{dep} patch"), &[s("git"), s("-C"), s(&target), s("apply"), s("--reverse"), s("--check"), s(p)], None, remedy);
    }
    if !cfg.opencode_dir.join("node_modules").is_dir() {
        errors.push(format!("OpenCode dependencies missing; run `harness2 setup {bench}`"));
    }
    if [&cfg.solver_model_oc, &cfg.improver_model].iter().any(|m| m.starts_with("google-vertex"))
        || (judge && (bench == "wb" || cfg.judge_model.starts_with("claude") || cfg.judge_model.starts_with("gemini")))
    {
        probe(&mut errors, "Google Application Default Credentials", &[s("gcloud"), s("auth"), s("application-default"), s("print-access-token")], None, remedy);
    }
    match bench.as_str() {
        "lab" => {
            for b in ["pandoc", "pdftoppm"] {
                if util::which(b).is_none() {
                    errors.push(format!("{b} missing; install pandoc and poppler-utils (README.md)"));
                }
            }
            probe(&mut errors, "LAB document and judge dependencies", &[s(cfg.lab_python()), s("-c"), s("import evaluation.scoring; import google.auth, anthropic")], Some(&repo), remedy);
        }
        "jb" if judge => {
            probe(&mut errors, "JobBench document and judge dependencies", &[s(cfg.jb_judge_python()), s("-c"), s("import pandas, openpyxl, docx, mammoth, pdfplumber, pptx, openai")], None, remedy);
            if cfg.jb_judge_api_base.is_empty() {
                errors.push("Set GOOGLE_CLOUD_PROJECT or HARNESS2_JUDGE_BASE_URL for JobBench judging".into());
            }
            if !cfg.judge_model.is_empty() && cfg.jb_judge_api_key.is_empty() && !cfg.judge_model.starts_with("gemini") {
                errors.push("A non-Gemini JobBench judge requires HARNESS2_JUDGE_BASE_URL and HARNESS2_JUDGE_API_KEY".into());
            }
        }
        "wb" => {
            probe(&mut errors, "Docker daemon", &[s("docker"), s("info"), s("--format"), s("{{.ServerVersion}}")], None, remedy);
            if adapter.spec().name == "wb-office" {
                probe(&mut errors, "WorkBuddy document rendering", &[s(crate::benchmarks::wb_render::render_python()), s("-c"), s("import openpyxl, pptx")], None, "install openpyxl and python-pptx for the renderer interpreter (HARNESS2_RENDER_PYTHON)");
            }
            probe(
                &mut errors,
                "WorkBuddy runtime",
                &[s(repo.join(".venv/bin/python")), s("-c"), s("import harbor; from workbuddy_bench.runner.harness_adapters import HARNESS_ADAPTERS; assert all(k in HARNESS_ADAPTERS for k in ('opencode', 'oc-harness', 'codex-harness'))")],
                Some(&repo),
                remedy,
            );
            for (role, expected) in [("solver", cfg.solver_model_oc.trim_start_matches("google-vertex/").to_string()), ("judge", cfg.judge_model.clone())] {
                let p = repo.join("configs/models/harness2").join(format!("{role}.yaml"));
                match std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                    Some(v) if v["model"]["name"].as_str() == Some(&format!("google/{expected}")) => {}
                    Some(_) => errors.push(format!("WorkBuddy {role} configuration is stale; rerun setup")),
                    None => errors.push(format!("WorkBuddy {role} configuration missing; rerun setup")),
                }
            }
        }
        _ => {}
    }
    errors
}

/// Fail before a caller spends anything when this benchmark is not ready to run.
pub fn require_ready(adapter: &dyn BenchmarkAdapter, judge: bool) -> Result<()> {
    let errors = check(adapter, judge);
    if !errors.is_empty() {
        return Err(anyhow!(ConfigError(format!("Setup incomplete; nothing has run yet:\n  - {}", errors.join("\n  - ")))));
    }
    Ok(())
}

#[derive(Args)]
pub struct CheckArgs {
    #[command(flatten)]
    pub cell: CellArgs,
}

pub fn check_cmd(a: CheckArgs) -> Result<i32> {
    let (errors, n) = match common::make_adapter(&a.cell, None) {
        Ok(ad) => {
            let e = check(ad.as_ref(), true);
            let n = ad.tasks("all").map(|t| t.len()).unwrap_or(0);
            (e, n)
        }
        Err(e) => (vec![e.to_string()], 0),
    };
    for e in &errors {
        println!("FAIL: {e}");
    }
    if !errors.is_empty() {
        println!("Setup incomplete: {} issue(s). See docs/BENCHMARKS.md.", errors.len());
        return Ok(1);
    }
    println!("Ready: {}/{}; {n} tasks. Model access is not exercised; the first real run is the first model call.", a.cell.bench, a.cell.substrate);
    Ok(0)
}

#[derive(Args)]
pub struct BridgeArgs {
    /// Vertex model ID to serve (default: HARNESS2_CODEX_MODEL)
    #[arg(long)]
    pub model: Option<String>,
    /// interface to bind; use 0.0.0.0 so WorkBuddy containers can reach it
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// default: the saved port, else a free one
    #[arg(long, default_value_t = 0)]
    pub port: u16,
}

const LITELLM_CONFIG: &str = "model_list:\n  - model_name: {model}\n    litellm_params:\n      model: vertex_ai/{model}\n      vertex_project: os.environ/GOOGLE_CLOUD_PROJECT\n      vertex_location: os.environ/GOOGLE_CLOUD_LOCATION\nlitellm_settings:\n  drop_params: true\ngeneral_settings:\n  master_key: os.environ/HARNESS2_BRIDGE_KEY\n";

/// Serve a Responses-API endpoint on Vertex AI for the Codex executor (via LiteLLM).
pub fn bridge_cmd(a: BridgeArgs) -> Result<i32> {
    let cfg = config::get();
    let model = a.model.unwrap_or_else(|| cfg.solver_model_cx.clone());
    let litellm = util::which("litellm").ok_or_else(|| anyhow!("LiteLLM is not installed; run: pip install 'litellm[proxy]==1.95.0'"))?;
    if cfg.vertex_project.is_empty() {
        bail!("Set GOOGLE_CLOUD_PROJECT in .env");
    }
    let root = cfg.dependencies.join("codex");
    let path = root.join("bridge.json");
    let saved: Value = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(json!({}));
    let port = if a.port != 0 {
        a.port
    } else if let Some(p) = saved["port"].as_u64() {
        p as u16
    } else {
        std::net::TcpListener::bind((a.host.as_str(), 0))?.local_addr()?.port()
    };
    let key = match saved["key"].as_str() {
        Some(k) => k.to_string(),
        None => {
            use base64::Engine;
            use std::io::Read;
            let mut buf = [0u8; 18];
            std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf)
        }
    };
    util::write(&path, format!("{}\n", json!({"port": port, "key": key})))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    let conf = root.join("bridge.yaml");
    util::write(&conf, LITELLM_CONFIG.replace("{model}", &model))?;
    println!("Add these to .env, then run `harness2 setup <bench> --substrate cx`:");
    println!("HARNESS2_CODEX_BASE_URL=http://127.0.0.1:{port}/v1\nHARNESS2_CODEX_API_KEY={key}\nHARNESS2_CODEX_MODEL={model}");
    if a.host != "127.0.0.1" {
        println!("HARNESS2_CODEX_CONTAINER_URL=http://host.docker.internal:{port}/v1");
    }
    let st = Command::new(litellm)
        .args(["--config", &s(&conf), "--host", &a.host, "--port", &port.to_string()])
        .env("HARNESS2_BRIDGE_KEY", &key)
        .env("GOOGLE_CLOUD_PROJECT", &cfg.vertex_project)
        .env("GOOGLE_CLOUD_LOCATION", &cfg.vertex_region)
        .status()?;
    Ok(st.code().unwrap_or(1))
}
