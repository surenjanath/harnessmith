//! The OpenCode agent framework as a solver substrate.
//!
//! Carries the c1..c8 surface (OPENCODE_SURFACE), the three install modes, the bun-run CLI
//! contract, and the one opencode event-stream trajectory renderer. The install modes are
//! different delivery channels for the always-on text and are not interchangeable.
//!
//! ```text
//!     workspace_agents_md   JB   AGENTS.md append + .opencode/{skills,agent} + scripts
//!     home_dir_jinja        WB   _composed/prompt.j2 + a component map for the container agent
//!     workspace_config      LAB  render composition + ENV_DELTA + the opencode.json tool gate
//! ```

use super::{cap_rendered, clip, pystr, Install};
use crate::config;
use crate::core::runmark;
use crate::core::tree::{self, SurfaceSpec};
use crate::util;
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn opencode_surface() -> SurfaceSpec {
    SurfaceSpec {
        name: "opencode",
        always_on: vec!["systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md"],
        always_on_max: 24_000,
        always_on_cap_reason: "every token here competes with the agent's own output budget",
        skill_dir: "skills",
        skill_name_rule: "must_match",
        trigger_hint: "Fix it by adding a trigger line naming `{name}` to systemprompt.md or guardrails.md, e.g. \"When <the situation this skill handles>, load the `{name}` skill.\"",
        autotrigger_file: "guardrails.md",
        trigger_line: "- When {desc}, load the `{name}` skill.\n",
        subagent_requires_mode: true,
        plugin_dir: Some("plugin"),
        hooks_file: None,
        hook_events: vec![],
        tool_dir: None,
        compile_skill_scripts: false,
        reject_empty_always_on: false,
        stubs: vec![
            ("scripts/README.md", "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. Add one by emitting scripts/<name>.py. -->\n"),
            ("sub_agents/README.md", "<!-- c5: delegated subagents, invoked with the `task` tool in their OWN context. Add one as sub_agents/<name>/AGENT.md with frontmatter `mode: subagent` and `description:`. -->\n"),
            ("plugin/README.md", "<!-- c8: OpenCode plugins. Acts INSIDE the tool loop -- the only lever that can enforce something the model cannot skip. Add one as plugin/<name>.ts. -->\n"),
            ("skills/README.md", "<!-- c6: reusable workflows loaded on demand via the `skill` tool. Add one as skills/<name>/SKILL.md with frontmatter `name:` matching the directory and a `description:`. -->\n"),
        ],
    }
}

pub const OUTPUT_TOKEN_MAX: u64 = 64_000;
pub const OC_INSTALL_MODES: &[&str] = &["workspace_agents_md", "home_dir_jinja", "workspace_config"];
const JINJA_PLACEHOLDER: &str = "{{ instruction }}";

/// Enforce the Jinja invariant for a composed home_dir_jinja template.
pub fn assert_jinja_safe(composed: &str) -> Result<()> {
    let n = composed.matches(JINJA_PLACEHOLDER).count();
    if n != 1 {
        bail!(
            "composed template must contain the '{{{{ instruction }}}}' placeholder exactly once (found {n}); Harbor renders it with StrictUndefined and substitutes the task instruction there, so a literal placeholder in always-on text would inject the task twice"
        );
    }
    let without = composed.replacen(JINJA_PLACEHOLDER, "", 1);
    if let Some(m) = Regex::new(r"\{\{|\{%|\{#").unwrap().find(&without) {
        bail!(
            "always-on text or a skill/subagent description contains a Jinja brace pair ({:?}); the composed prompt template renders with StrictUndefined, so any braces other than the one '{{{{ instruction }}}}' placeholder kill the job pre-trial (and '{{#' comments delete text silently). Use a single brace, or words.",
            m.as_str()
        );
    }
    if !composed.trim_end().ends_with(JINJA_PLACEHOLDER) {
        bail!("composed template must END with the instruction placeholder (StrictUndefined render contract)");
    }
    Ok(())
}

pub fn frontmatter_desc(body: &str) -> String {
    let fm = util::take_chars(body, 600);
    Regex::new(r"(?m)^description:\s*(.+)$")
        .unwrap()
        .captures(&fm)
        .map(|c| c[1].trim().to_string())
        .unwrap_or_else(|| "(no description)".into())
}

/// One "- name: description" line per component file `<dir>/*/<file>`, or "".
pub fn component_index(staged: &Path, dir: &str, file: &str) -> String {
    let mut found: Vec<PathBuf> = std::fs::read_dir(staged.join(dir))
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path().join(file)).filter(|p| p.is_file()).collect())
        .unwrap_or_default();
    found.sort_by_key(|p| p.parent().unwrap().file_name().unwrap().to_os_string());
    found
        .iter()
        .map(|p| {
            format!(
                "- `{}`: {}",
                p.parent().unwrap().file_name().unwrap().to_string_lossy(),
                frontmatter_desc(&util::read_lossy(p).unwrap_or_default())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the Jinja template that carries the always-on harness into a WB run.
pub fn compose_prompt(staged: &Path) -> String {
    let mut parts: Vec<String> = opencode_surface()
        .always_on
        .iter()
        .filter_map(|f| util::read_lossy(&staged.join(f)).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()))
        .collect();
    let idx = component_index(staged, "skills", "SKILL.md");
    if !idx.is_empty() {
        parts.push(format!("# AVAILABLE SKILLS\nLoad with the `skill` tool when the situation matches:\n{idx}"));
    }
    let idx = component_index(staged, "sub_agents", "AGENT.md");
    if !idx.is_empty() {
        parts.push(format!("# AVAILABLE SUBAGENTS\nDelegate with the `task` tool when the situation matches:\n{idx}"));
    }
    let mut scripts: Vec<String> = std::fs::read_dir(staged.join("scripts"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.ends_with(".py"))
                .collect()
        })
        .unwrap_or_default();
    scripts.sort();
    if !scripts.is_empty() {
        parts.push(format!(
            "# HELPER SCRIPTS\nAvailable at `~/harness_scripts/`: {}\nRun with `python ~/harness_scripts/<name>.py`.",
            scripts.join(", ")
        ));
    }
    parts.push(JINJA_PLACEHOLDER.into());
    parts.join("\n\n---\n\n")
}

/// OpenCode behind the substrate interface; the launcher body is benchmark-supplied.
pub struct OpencodeSubstrate;

impl OpencodeSubstrate {
    pub const NAME: &'static str = "oc";

    pub fn surface(&self) -> SurfaceSpec {
        opencode_surface()
    }

    /// Stage the editable tree as substrate-native files, per `install_mode`. `ctx` carries
    /// benchmark-owned `file:<rel>` entries for integration files that ride along.
    pub fn materialise(&self, tree_dir: &Path, install_mode: &str, ctx: &BTreeMap<String, String>) -> Result<Install> {
        if !OC_INSTALL_MODES.contains(&install_mode) {
            bail!(
                "install_mode {install_mode:?} is not an opencode mode; this substrate supports {OC_INSTALL_MODES:?} (codex_root belongs to CodexSubstrate)"
            );
        }
        if !tree::is_tree(&opencode_surface(), tree_dir) {
            bail!("{} is not an opencode harness tree (no always-on file present)", tree_dir.display());
        }
        let staged = tempfile::Builder::new().prefix(&format!("adapt-oc-{install_mode}-")).tempdir()?.keep();
        let res: Result<()> = (|| {
            util::copy_tree(tree_dir, &staged)?;
            for (key, body) in ctx {
                if let Some(rel) = key.strip_prefix("file:") {
                    util::write(&staged.join(tree::normalize_rel(rel)?), body)?;
                }
            }
            if install_mode == "home_dir_jinja" {
                let composed = compose_prompt(&staged);
                let tg = staged.join("tool_descriptions/tool_guidance.md");
                if tg.is_file() {
                    std::fs::rename(&tg, staged.join("tool_guidance.md"))?;
                    util::rmtree(&staged.join("tool_descriptions"));
                }
                assert_jinja_safe(&composed)?;
                util::write(&staged.join("_composed/prompt.j2"), composed)?;
            }
            Ok(())
        })();
        if let Err(e) = res {
            util::rmtree(&staged);
            return Err(e);
        }
        let mut extras = BTreeMap::new();
        extras.insert("install_mode".into(), install_mode.into());
        Ok(Install { staging_dir: staged, fingerprint: runmark::fingerprint(Some(tree_dir)), extras })
    }

    /// The reference argv. The upstream checkout runs from source through bun.
    #[allow(clippy::too_many_arguments)]
    pub fn run_argv(&self, prompt: &str, workspace: &Path, model: &str, title: &str, variant: Option<&str>, agent: &str, session: Option<&str>) -> Vec<String> {
        let cfg = config::get();
        let mut argv = vec![
            cfg.bun.to_string_lossy().to_string(),
            "run".into(),
            "--cwd".into(),
            cfg.opencode_dir.join("packages/opencode").to_string_lossy().to_string(),
            "--conditions=browser".into(),
            "src/index.ts".into(),
            "run".into(),
            prompt.into(),
            "--dir".into(),
            workspace.to_string_lossy().to_string(),
            "--model".into(),
            model.into(),
            "--title".into(),
            title.into(),
            "--format".into(),
            "json".into(),
        ];
        if !agent.is_empty() {
            argv.extend(["--agent".into(), agent.into()]);
        }
        if let Some(v) = variant.filter(|v| !v.is_empty()) {
            argv.extend(["--variant".into(), v.into()]);
        }
        if let Some(s) = session {
            argv.extend(["--session".into(), s.into()]);
        }
        argv
    }

    /// The reference permission fence and output-token ceiling.
    pub fn run_env(&self, workspace: &Path, output_token_max: u64, extra_allowed: &[PathBuf], model: &str) -> BTreeMap<String, String> {
        let w = workspace.to_string_lossy().to_string();
        let mut allow = serde_json::Map::new();
        allow.insert("*".into(), json!("deny"));
        allow.insert(w.clone(), json!("allow"));
        allow.insert(format!("{w}/**"), json!("allow"));
        for r in extra_allowed {
            let r = r.to_string_lossy().to_string();
            allow.insert(r.clone(), json!("allow"));
            allow.insert(format!("{r}/**"), json!("allow"));
        }
        let m = if model.is_empty() { config::get().solver_model_oc.clone() } else { model.to_string() };
        let mut env = config::scrub_env(None, config::provider_keys(&m));
        env.insert("OPENCODE_PERMISSION".into(), json!({"external_directory": Value::Object(allow)}).to_string());
        env.insert("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX".into(), output_token_max.to_string());
        env
    }

    pub fn read_trajectory(&self, stream_path: &Path, obs_trunc: usize, max_chars: usize) -> String {
        render_events(stream_path, obs_trunc, max_chars, 400)
    }

    /// Checks the OpenCode checkout and bun exist; starts nothing.
    pub fn preflight(&self) -> Result<()> {
        let cfg = config::get();
        if !cfg.opencode_dir.join("packages/opencode/src/index.ts").is_file() {
            bail!(
                "OpenCode checkout missing at {} (no packages/opencode/src/index.ts): run `harness2 setup <benchmark>` to fetch the pinned checkout at {} (v1.14.18).",
                cfg.opencode_dir.display(),
                &config::pin("opencode")[..12]
            );
        }
        if !cfg.bun.is_file() {
            bail!(
                "bun not found at {}: install bun on PATH or at ~/.bun/bin/bun (the upstream OpenCode is run from source through bun).",
                cfg.bun.display()
            );
        }
        Ok(())
    }
}

/// Render OpenCode's `--format json` JSONL event stream into the shared grammar.
pub fn render_events(path: &Path, obs_trunc: usize, max_chars: usize, args_trunc: usize) -> String {
    let raw = match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => return format!("(failed to read trajectory: {e})"),
    };
    let mut lines = vec![];
    for ln in raw.lines() {
        let ln = ln.trim();
        if !ln.starts_with('{') {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(ln) else { continue };
        if !ev.is_object() {
            continue;
        }
        if let Some(err) = ev.get("error").filter(|e| e.is_object()) {
            let msg = err
                .get("data")
                .and_then(|d| d.get("message"))
                .map(|m| pystr(Some(m)))
                .filter(|s| !s.is_empty())
                .or_else(|| err.get("message").map(|m| pystr(Some(m))).filter(|s| !s.is_empty()))
                .unwrap_or_else(|| util::take_chars(&err.to_string(), 300));
            lines.push(format!("[ERROR] {}", clip(&msg, 300)));
            continue;
        }
        let part = ev.get("part").filter(|p| p.is_object()).unwrap_or(&ev);
        let ptype = part.get("type").or_else(|| ev.get("type")).and_then(|t| t.as_str()).unwrap_or("");
        match ptype {
            "text" | "reasoning" => {
                let tag = if ptype == "text" { "ASSISTANT" } else { "THINKING" };
                let txt = part.get("text").or_else(|| part.get("content"));
                if let Some(t) = txt.and_then(|t| t.as_str()).filter(|t| !t.trim().is_empty()) {
                    lines.push(format!("[{tag}] {}", clip(t, obs_trunc)));
                }
            }
            "tool" | "tool_use" | "tool_call" | "tool-invocation" => {
                let name = [part.get("tool"), ev.get("tool"), part.get("name"), part.get("toolInvocation").and_then(|t| t.get("toolName"))]
                    .into_iter()
                    .flatten()
                    .map(|v| pystr(Some(v)))
                    .find(|s| !s.is_empty())
                    .unwrap_or_else(|| "tool".into());
                let state = part.get("state").filter(|s| s.is_object()).unwrap_or(part);
                let status = state.get("status").and_then(|s| s.as_str()).unwrap_or("");
                let tin = state.get("input").or_else(|| part.get("input")).or_else(|| part.get("args")).filter(|v| !v.is_null());
                let tout = state.get("output").or_else(|| state.get("result")).or_else(|| part.get("output")).filter(|v| !v.is_null());
                let args = tin.map(|v| v.to_string()).unwrap_or_default();
                lines.push(format!("[ACTION] {name}({})", clip(&args, args_trunc)));
                if status == "error" {
                    lines.push(format!("[OBSERVATION ERROR] {}", clip(&pystr(state.get("error")), obs_trunc)));
                } else if let Some(o) = tout {
                    lines.push(format!("[OBSERVATION] {}", clip(&pystr(Some(o)), obs_trunc)));
                }
            }
            _ => {}
        }
    }
    cap_rendered(&lines, max_chars)
}
