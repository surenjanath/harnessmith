//! The native `api` proposer channel: one improver/selector session in a context workspace.
//!
//! Mirrors the OpenCode channel's contract: the mission is written to MISSION.md, the model
//! works on the files in place (that is how an edited harness comes back), and the answer is
//! `output/answer.md` when written, else the final message.

use super::agent::{self, AgentConfig};
use crate::core::spec::ChannelSpec;
use crate::util;
use anyhow::Result;
use std::path::Path;
use std::time::Duration;

pub struct SessionResult {
    pub text: String,
    pub cost_usd: f64,
    pub error: String,
}

pub const IMPROVER_SYSTEM: &str = "You are an autonomous agent working inside a workspace directory on disk. Your instructions are in MISSION.md and your evidence is in the files it points to. Use the file tools to read what you need and to edit files in place -- edits on disk are how your work is delivered. Read before you write, keep edits minimal and deliberate, and do not invent files the mission did not ask for. When you are done, reply with the requested output as your final message (or call `finish` with it).";

pub fn run_improver_session(spec: &ChannelSpec, cwd: &Path, mission: &str, system: &str, wall: u64) -> Result<SessionResult> {
    util::write(&cwd.join("MISSION.md"), format!("{}{mission}", if system.is_empty() { String::new() } else { format!("{system}\n\n---\n\n") }))?;
    std::fs::create_dir_all(cwd.join("output"))?;
    let mut cfg = AgentConfig::new(&spec.model, IMPROVER_SYSTEM, cwd);
    cfg.improver = true;
    cfg.max_steps = spec.steps.max(1);
    cfg.wall = Duration::from_secs(wall.max(60));
    cfg.max_tokens = match spec.effort.as_str() {
        "low" => 4096,
        "medium" => 8192,
        _ => 16384,
    };
    cfg.rescue_hint = "Finish now: complete any edit in context/harness/ you have started, then reply with your final <notes> block (or the requested output). Unfinished work is lost when the budget ends.".into();
    let prompt = "Read MISSION.md first. Do what it asks. Reply with the requested output as your final message AND save it verbatim to output/answer.md.";
    let out = agent::run(&cfg, prompt)?;
    let ans = util::read_lossy(&cwd.join("output/answer.md")).unwrap_or_default();
    let text = if !ans.trim().is_empty() { ans } else { out.final_text };
    Ok(SessionResult { text, cost_usd: out.cost_usd, error: out.error })
}
