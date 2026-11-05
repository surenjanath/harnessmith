//! JobBench OpenCode launcher: stage a task workspace, run OpenCode in it, keep the artifacts.
//!
//! Invoked as `harness2 jb-oc-runner` with the same environment contract as the reference
//! `jb_opencode_runner.py` (MODEL_ID, LABEL, HARNESS_RENDER_DIR, TASK_IDS, ...).

use crate::util;
use anyhow::{bail, Result};
use serde_json::json;
use sha1::{Digest, Sha1};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const TEMP_BASE: &str = "/tmp/hevo_ws";
const EVENT_PREFIX: &str = "{\"type\":";
const PROMPT: &str = "=== TASK FOLDER ===\n{task_folder}\n\n=== INSTRUCTIONS ===\n1. Read the TASK_INSTRUCTIONS.txt file in the task folder above\n2. Based on the Reference Files section in TASK_INSTRUCTIONS.txt, read the corresponding files from the same task folder using appropriate tools.\n3. Complete the task as specified in TASK_INSTRUCTIONS.txt\n4. Only save the final deliverables to the output directory specified below. Do not save any intermediate or temporary files.\n\n=== OUTPUT DIRECTORY ===\n{output_dir}\n\nIMPORTANT:\n- All reference files are in the task folder: {task_folder}\n- Only save the final deliverables to the output directory {output_dir}. Do not save any intermediate or temporary files.\n- You MUST only access files within {workspace} or search online for new reference files if you find needed. Do NOT access any files or directories in this system outside of this path.\n- If you encounter ambiguous or conflicting information, analyze the conflict, explain your reasoning, and justify the approach you choose.\n- If a file cannot be read directly (e.g., .xlsx, .docx, .db, .pptx), use appropriate tools, MCP servers, or code to extract and process its contents.";

struct Settings {
    task_ids: Vec<String>,
    label: String,
    render_dir: PathBuf,
    tasks_base: PathBuf,
    opencode_dir: PathBuf,
    model_id: String,
    agent_name: String,
    variant: String,
    agent_steps: String,
    timeout_per_task: String,
    max_retries: u32,
    log_dir: PathBuf,
}

fn env(n: &str) -> String {
    std::env::var(n).unwrap_or_default()
}

fn settings() -> Result<Settings> {
    let missing: Vec<&str> = ["MODEL_ID", "LABEL", "HARNESS_RENDER_DIR", "TASK_IDS"].into_iter().filter(|n| env(n).is_empty()).collect();
    if !missing.is_empty() {
        bail!("[ERROR] unset: {}", missing.join(", "));
    }
    let render_dir = PathBuf::from(env("HARNESS_RENDER_DIR"));
    if !render_dir.join(".opencode").is_dir() {
        bail!("[ERROR] HARNESS_RENDER_DIR has no .opencode/: {}", render_dir.display());
    }
    let cfg = crate::config::get();
    let opencode_dir = if env("OPENCODE_DIR").is_empty() { cfg.opencode_dir.clone() } else { PathBuf::from(env("OPENCODE_DIR")) };
    if !opencode_dir.join("packages/opencode").is_dir() {
        bail!("[ERROR] OpenCode not installed at {}", opencode_dir.display());
    }
    let tasks_base = if env("TASKS_BASE_DIR").is_empty() { cfg.jb_repo().join("dataset/main") } else { PathBuf::from(env("TASKS_BASE_DIR")) };
    Ok(Settings {
        task_ids: env("TASK_IDS").split_whitespace().map(|s| s.to_string()).collect(),
        label: env("LABEL"),
        render_dir,
        log_dir: tasks_base.join(".harness2-logs"),
        tasks_base,
        opencode_dir,
        model_id: env("MODEL_ID"),
        agent_name: env("AGENT_NAME"),
        variant: env("VARIANT"),
        agent_steps: env("AGENT_STEPS"),
        timeout_per_task: if env("TIMEOUT_PER_TASK").is_empty() { "3600".into() } else { env("TIMEOUT_PER_TASK") },
        max_retries: env("MAX_RETRIES").parse().unwrap_or(2),
    })
}

fn hms() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}
fn stamp() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}
fn nonempty(p: &Path) -> bool {
    p.is_dir() && std::fs::read_dir(p).map(|mut d| d.next().is_some()).unwrap_or(false)
}
fn log(f: &Path, line: &str) {
    println!("{line}");
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(f) {
        let _ = writeln!(fh, "{line}");
    }
}

/// This rollout's npm prefix, Python user base and temp dir; short for AF_UNIX socket paths.
fn sandbox(ws: &Path) -> PathBuf {
    let h = hex::encode(Sha1::digest(ws.file_name().unwrap().to_string_lossy().as_bytes()));
    ws.parent().unwrap().join(format!("sb-{}", &h[..12]))
}

fn permission(ws: &Path) -> String {
    let w = ws.to_string_lossy();
    let mut m = serde_json::Map::new();
    for (k, v) in [
        ("*".to_string(), "deny"),
        ("/tmp".into(), "allow"),
        ("/tmp/*".into(), "allow"),
        ("/tmp/**".into(), "allow"),
        (format!("{TEMP_BASE}/*"), "allow"),
        (format!("{TEMP_BASE}/**"), "deny"),
        (format!("{}/**", sandbox(ws).display()), "allow"),
        (format!("{w}/**"), "allow"),
    ] {
        m.insert(k, json!(v));
    }
    json!({"external_directory": m}).to_string()
}

fn stage_sandbox(ws: &Path) -> Result<()> {
    let sb = sandbox(ws);
    util::rmtree(&sb);
    for n in ["npm-global", "python-user", "tmp"] {
        std::fs::create_dir_all(sb.join(n))?;
    }
    Ok(())
}

fn reset_task(task_dir: &Path, ws: &Path) -> Result<()> {
    util::rmtree(&ws.join("task_folder"));
    util::rmtree(&ws.join("output"));
    util::copy_tree(task_dir, &ws.join("task_folder"))?;
    std::fs::create_dir_all(ws.join("output"))?;
    Ok(())
}

fn stage(task_dir: &Path, render_dir: &Path, ws: &Path) -> Result<()> {
    util::rmtree(ws);
    std::fs::create_dir_all(ws.join("output"))?;
    util::copy_tree(task_dir, &ws.join("task_folder"))?;
    util::copy_tree(&render_dir.join(".opencode"), &ws.join(".opencode"))?;
    if render_dir.join("AGENTS.md").is_file() {
        std::fs::copy(render_dir.join("AGENTS.md"), ws.join("AGENTS.md"))?;
    }
    if render_dir.join("scripts").is_dir() {
        util::copy_tree(&render_dir.join("scripts"), &ws.join("scripts"))?;
    }
    Ok(())
}

fn attempt(cfg: &Settings, ws: &Path, traj: &Path, log_file: &Path, task_name: &str, n: u32) -> Result<i32> {
    if !cfg.agent_steps.is_empty() && !ws.join("opencode.json").is_file() {
        let steps: serde_json::Value = serde_json::from_str(&cfg.agent_steps)?;
        util::write(&ws.join("opencode.json"), format!("{}\n", json!({"agent": {"build": {"steps": steps}}})))?;
    }
    let prompt = PROMPT
        .replace("{task_folder}", &ws.join("task_folder").to_string_lossy())
        .replace("{output_dir}", &ws.join("output").to_string_lossy())
        .replace("{workspace}", &ws.to_string_lossy());
    let title = format!("{task_name} | {} | a{n}", cfg.label);
    // GNU `timeout` is missing on macOS unless coreutils is installed as `gtimeout` (upstream #77).
    let tbin = ["timeout", "gtimeout"].into_iter().find(|b| util::which(b).is_some()).ok_or_else(|| {
        anyhow::anyhow!("neither `timeout` nor `gtimeout` is on PATH; the OpenCode runner needs one to enforce its time limit (macOS: `brew install coreutils`)")
    })?;
    let mut c = Command::new(tbin);
    c.args(["--kill-after=30s", &cfg.timeout_per_task, "bun", "run", "--cwd"])
        .arg(cfg.opencode_dir.join("packages/opencode"))
        .args(["--conditions=browser", "src/index.ts", "run", &prompt, "--dir"])
        .arg(ws);
    if !cfg.agent_name.is_empty() {
        c.args(["--agent", &cfg.agent_name]);
    }
    if !cfg.variant.is_empty() {
        c.args(["--variant", &cfg.variant]);
    }
    c.args(["--model", &cfg.model_id, "--title", &title, "--format", "json"]);
    let sb = sandbox(ws);
    let path = env("PATH");
    c.current_dir(&cfg.opencode_dir)
        .env("NPM_CONFIG_PREFIX", sb.join("npm-global"))
        .env("PYTHONUSERBASE", sb.join("python-user"))
        .env("TMPDIR", sb.join("tmp"))
        .env("PATH", format!("{}:{}:{path}", sb.join("npm-global/bin").display(), sb.join("python-user/bin").display()))
        .env("OPENCODE_PERMISSION", permission(ws));
    let raw = traj.with_file_name(format!(".{task_name}_a{n}.stdout.{}", std::process::id()));
    let out = std::fs::File::create(&raw)?;
    let err = std::fs::OpenOptions::new().create(true).append(true).open(log_file)?;
    c.stdin(Stdio::null()).stdout(out).stderr(err);
    let st = c.status()?;
    let code = st.code().unwrap_or_else(|| {
        use std::os::unix::process::ExitStatusExt;
        -st.signal().unwrap_or(9)
    });
    let body = util::read_lossy(&raw).unwrap_or_default();
    let events: String = body.split_inclusive('\n').filter(|l| l.starts_with(EVENT_PREFIX)).collect();
    std::fs::write(traj, events)?;
    let _ = std::fs::remove_file(&raw);
    Ok(code)
}

fn note_exit(traj: &Path, log_file: &Path, code: i32, limit: &str, kept: bool) {
    if code == 0 {
        return;
    }
    let span = if limit.chars().all(|c| c.is_ascii_digit()) { format!("{limit} s") } else { limit.to_string() };
    let why = match code {
        124 => format!("was stopped at the {span} time limit (exit 124)"),
        -9 | 137 => format!("was killed with SIGKILL ({code}), after the {span} time limit or by the system"),
        c if c < 0 => format!("was killed by signal {}", -c),
        c => format!("exited with status {c}"),
    };
    let outcome = if kept { "files already in the output directory were kept as the deliverable" } else { "the output directory was empty" };
    let message = format!("Rollout runner: the executor {why}; {outcome}.");
    let ev = json!({"type": "error", "timestamp": chrono::Utc::now().timestamp_millis(), "error": {"name": "RunnerExit", "data": {"message": message}}});
    if let Ok(mut fh) = std::fs::OpenOptions::new().append(true).create(true).open(traj) {
        let _ = write!(fh, "\n{ev}\n");
    }
    log(log_file, &format!("[{}] EXIT {code}: {message}", hms()));
}

fn run_one(cfg: &Settings, task_id: &str) -> Result<bool> {
    let parent = cfg.tasks_base.join(task_id);
    let task_dir = parent.join("task_folder");
    let task_name = task_id.replace('/', "_");
    let final_out = parent.join("model_output").join(&cfg.label);
    let traj_dir = parent.join("model_traj").join(&cfg.label);
    let log_file = cfg.log_dir.join(format!("hevo_{task_name}_{}_{}.log", cfg.label, stamp()));
    if !task_dir.is_dir() {
        log(&log_file, &format!("[ERROR] no task_folder: {}", task_dir.display()));
        return Ok(false);
    }
    if nonempty(&final_out) {
        println!("[{}] SKIP (done): {task_id} ({})", hms(), cfg.label);
        return Ok(true);
    }
    std::fs::create_dir_all(&traj_dir)?;
    let ws = PathBuf::from(TEMP_BASE).join(format!("{task_name}_{}_{}", cfg.label, std::process::id()));
    let output_dir = ws.join("output");
    stage(&task_dir, &cfg.render_dir, &ws)?;
    stage_sandbox(&ws)?;
    for n in 1..=cfg.max_retries {
        if n > 1 {
            reset_task(&task_dir, &ws)?;
            stage_sandbox(&ws)?;
        }
        let traj = traj_dir.join(format!("{task_name}_{}_attempt{n}_{}.jsonl", cfg.label, stamp()));
        log(&log_file, &format!("[{}] run {task_id} ({}) attempt {n} -> {}", hms(), cfg.label, traj.display()));
        let code = attempt(cfg, &ws, &traj, &log_file, &task_name, n)?;
        note_exit(&traj, &log_file, code, &cfg.timeout_per_task, nonempty(&output_dir));
        if nonempty(&output_dir) {
            break;
        }
        if code == 124 || code == 137 {
            log(&log_file, &format!("[{}] TIMEOUT {task_id} attempt {n}", hms()));
        }
    }
    let delivered = nonempty(&output_dir);
    if delivered {
        std::fs::create_dir_all(&final_out)?;
        let mut items: Vec<PathBuf> = std::fs::read_dir(&output_dir)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        items.sort();
        for it in items {
            let name = it.file_name().unwrap().to_owned();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            if std::fs::rename(&it, final_out.join(&name)).is_err() {
                if it.is_dir() {
                    util::copy_tree(&it, &final_out.join(&name))?;
                } else {
                    std::fs::copy(&it, final_out.join(&name))?;
                }
            }
        }
        log(&log_file, &format!("[{}] output -> {}", hms(), final_out.display()));
    } else {
        log(&log_file, &format!("[{}] NO OUTPUT for {task_id} ({})", hms(), cfg.label));
    }
    util::rmtree(&ws);
    util::rmtree(&sandbox(&ws));
    Ok(delivered && nonempty(&final_out))
}

pub fn main() -> Result<i32> {
    let cfg = settings()?;
    println!("=== harness eval: LABEL={} MODEL={} AGENT={} ===", cfg.label, cfg.model_id, cfg.agent_name);
    println!("render: {}", cfg.render_dir.display());
    std::fs::create_dir_all(&cfg.log_dir)?;
    std::fs::create_dir_all(TEMP_BASE)?;
    let mut failed = vec![];
    for t in &cfg.task_ids {
        if !run_one(&cfg, t)? {
            failed.push(t.clone());
        }
    }
    println!("=== harness eval done: {} ===", cfg.label);
    if !failed.is_empty() {
        println!("no output for: {}", failed.join(", "));
        return Ok(1);
    }
    Ok(0)
}
