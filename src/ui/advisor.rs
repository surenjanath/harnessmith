//! Model advisor: what this machine can run, which models are installed where, how fast they
//! are, and the best role setup — with one-click presets and model downloads for Ollama.
//!
//! Hardware: CPU, RAM, NVIDIA GPUs (nvidia-smi), Apple Silicon unified memory. The memory budget
//! for a model is the largest GPU's VRAM, ~70% of unified memory on Apple Silicon, or ~50% of RAM
//! on CPU only. Speed is estimated from memory bandwidth / model size (decoding is memory-bound)
//! until a benchmark measures it.

use super::{blocking, ApiResult};
use crate::llm::{client, connections};
use crate::util;
use anyhow::Result;
use axum::Json;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// What this machine has, and how much memory a model may use.
pub fn hardware() -> Value {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (mut cpu, mut ram_gb, mut avail_gb) = (String::new(), 0.0f64, 0.0f64);
    if os == "linux" {
        let ci = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        cpu = ci.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).unwrap_or("").trim().to_string();
        let mi = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let kb = |k: &str| mi.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        ram_gb = kb("MemTotal:") / 1048576.0;
        avail_gb = kb("MemAvailable:") / 1048576.0;
    } else if os == "macos" {
        cpu = run("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
        ram_gb = run("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0) / 1073741824.0;
    }
    let mut gpus = vec![];
    if let Some(s) = run("nvidia-smi", &["--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]) {
        for l in s.lines() {
            let p: Vec<&str> = l.split(',').map(|x| x.trim()).collect();
            if p.len() >= 3 {
                gpus.push(json!({"name": p[0], "vram_gb": p[1].parse::<f64>().unwrap_or(0.0) / 1024.0, "used_gb": p[2].parse::<f64>().unwrap_or(0.0) / 1024.0, "kind": "nvidia"}));
            }
        }
    }
    let apple = os == "macos" && arch == "aarch64";
    if apple {
        gpus.push(json!({"name": format!("{} GPU (unified memory)", if cpu.is_empty() { "Apple Silicon".into() } else { cpu.clone() }), "vram_gb": ram_gb * 0.70, "kind": "apple"}));
    }
    let best_vram = gpus.iter().filter(|g| g["kind"] == "nvidia").filter_map(|g| g["vram_gb"].as_f64()).fold(0.0, f64::max);
    let (mode, budget, bandwidth) = if best_vram > 2.0 {
        ("gpu", best_vram * 0.92, 450.0)
    } else if apple {
        // M-series bandwidth ranges ~70-800 GB/s; Pro/Max chips are named in the brand string.
        let bw = if cpu.contains("Ultra") { 700.0 } else if cpu.contains("Max") { 400.0 } else if cpu.contains("Pro") { 200.0 } else { 100.0 };
        ("unified", ram_gb * 0.70, bw)
    } else {
        ("cpu", ram_gb * 0.50, 40.0)
    };
    json!({"os": os, "arch": arch, "cpu": cpu, "cores": cores, "ram_gb": round1(ram_gb), "ram_available_gb": round1(avail_gb), "gpus": gpus,
        "mode": mode, "budget_gb": round1(budget), "bandwidth_gbs": bandwidth,
        "summary": match mode { "gpu" => format!("GPU with {:.0} GB VRAM", best_vram), "unified" => format!("Apple Silicon, {:.0} GB unified memory", ram_gb), _ => format!("CPU only, {:.0} GB RAM ({} cores)", ram_gb, cores) }})
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

static PARAMS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*([bm])\b").unwrap());

fn parse_params(s: &str) -> Option<f64> {
    PARAMS.captures_iter(&s.replace(['-', '_', ':'], " ")).filter_map(|c| c[1].parse::<f64>().ok().map(|n| if c[2].eq_ignore_ascii_case("m") { n / 1000.0 } else { n })).filter(|n| *n > 0.05 && *n < 1000.0).last()
}

fn kind_of(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("embed") || n.contains("bge-") || n.contains("e5-") || n.contains("minilm") {
        "embed"
    } else if n.contains("vision") || n.contains("llava") || n.contains("-vl") {
        "vision"
    } else {
        "chat"
    }
}

/// Families that follow tool-calling instructions well at small sizes (rough prior).
fn family_bonus(name: &str) -> f64 {
    let n = name.to_lowercase();
    for (k, b) in [("qwen3", 1.3), ("qwen2.5", 1.15), ("gpt-oss", 1.25), ("llama3.3", 1.15), ("llama3.1", 1.05), ("mistral-small", 1.1), ("mistral-nemo", 1.0), ("gemma3", 1.05), ("phi4", 1.0), ("deepseek-r1", 0.85), ("llama3.2", 0.9), ("tinyllama", 0.5)] {
        if n.contains(k) {
            return b;
        }
    }
    0.9
}

static BENCH: Lazy<Mutex<BTreeMap<String, Value>>> = Lazy::new(|| {
    let p = crate::config::get().runs.join("bench.json");
    Mutex::new(std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default())
});

fn is_local(base: &str) -> bool {
    let b = base.to_lowercase();
    ["localhost", "127.", "0.0.0.0", "192.168.", "10.", "host.docker.internal", "[::1]"].iter().any(|h| b.contains(&format!("//{h}")))
}

/// Every model this machine can reach: detected local servers, saved connections, API ones.
fn inventory(hw: &Value) -> (Vec<Value>, Vec<Value>) {
    let reg = connections::load();
    let budget = hw["budget_gb"].as_f64().unwrap_or(0.0);
    let bw = hw["bandwidth_gbs"].as_f64().unwrap_or(40.0);
    let bench = BENCH.lock().unwrap().clone();
    let mut servers: Vec<Value> = vec![];
    let mut models: Vec<Value> = vec![];
    // Local servers: what discovery found, plus saved connections that point at local/LAN hosts.
    let mut seen_bases: Vec<String> = vec![];
    let mut local: Vec<(String, String, String, bool)> = vec![]; // (name, provider, base, saved)
    for c in reg.connections.iter().filter(|c| c.provider == "ollama" || is_local(&c.base())) {
        seen_bases.push(c.base().trim_end_matches("/v1").to_string());
        local.push((c.name.clone(), c.provider.clone(), c.base(), true));
    }
    for d in client::discover_local() {
        let base = d["base_url"].as_str().unwrap_or("").to_string();
        if !seen_bases.contains(&base.trim_end_matches("/v1").to_string()) {
            local.push((d["name"].as_str().unwrap_or("").into(), d["provider"].as_str().unwrap_or("").into(), base, false));
        }
    }
    for (name, provider, base, saved) in local {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_millis(800)).timeout_read(Duration::from_secs(6)).try_proxy_from_env(false).build();
        let root = base.trim_end_matches('/').trim_end_matches("/v1").to_string();
        let mut list: Vec<(String, Option<f64>, Option<f64>, String, String)> = vec![]; // name, size_gb, params_b, quant, family
        let mut ok = false;
        if provider == "ollama" {
            if let Some(v) = agent.get(&format!("{root}/api/tags")).call().ok().and_then(|r| r.into_string().ok()).and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
                ok = true;
                for m in v["models"].as_array().cloned().unwrap_or_default() {
                    let n = m["name"].as_str().unwrap_or("").to_string();
                    let d = &m["details"];
                    list.push((n.clone(), m["size"].as_f64().map(|b| b / 1e9), d["parameter_size"].as_str().and_then(parse_params).or_else(|| parse_params(&n)), d["quantization_level"].as_str().unwrap_or("").into(), d["family"].as_str().unwrap_or("").into()));
                }
            }
        } else if let Some(v) = agent.get(&format!("{}/models", base.trim_end_matches('/'))).call().ok().and_then(|r| r.into_string().ok()).and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
            ok = true;
            for m in v["data"].as_array().cloned().unwrap_or_default() {
                let n = m["id"].as_str().unwrap_or("").to_string();
                list.push((n.clone(), None, parse_params(&n), String::new(), String::new()));
            }
        }
        servers.push(json!({"name": name, "provider": provider, "base_url": base, "saved": saved, "reachable": ok, "models": list.len(), "kind": "local"}));
        for (n, size, params, quant, family) in list {
            // Size unknown (OpenAI-compatible list): ~0.6 GB per billion parameters at Q4.
            let size_gb = size.or(params.map(|p| p * 0.6)).unwrap_or(0.0);
            let need = size_gb * 1.15 + 1.0;
            let fits = if size_gb == 0.0 { "unknown" } else if need <= budget { "fits" } else if need <= budget * 1.25 { "tight" } else { "too big" };
            let id = format!("{name}/{n}");
            let measured = bench.get(&id).and_then(|b| b["tok_s"].as_f64());
            let est = if size_gb > 0.0 { (bw / size_gb * 0.6).min(250.0) } else { 0.0 };
            models.push(json!({"id": id, "server": name, "provider": provider, "saved": saved, "base_url": base, "name": n, "size_gb": round1(size_gb), "params_b": params,
                "quant": quant, "family": family, "kind": kind_of(&n), "fits": fits, "est_tok_s": round1(est), "measured_tok_s": measured, "local": true}));
        }
    }
    // API connections with a key: always "fit"; speed unknown until benchmarked.
    for c in reg.connections.iter().filter(|c| !(c.provider == "ollama" || is_local(&c.base())) && c.provider != "mock" && !c.key().is_empty() && !c.default_model.is_empty()) {
        servers.push(json!({"name": c.name, "provider": c.provider, "base_url": c.base(), "saved": true, "reachable": true, "models": 1, "kind": "api"}));
        let id = format!("{}/{}", c.name, c.default_model);
        models.push(json!({"id": id, "server": c.name, "provider": c.provider, "saved": true, "name": c.default_model, "kind": kind_of(&c.default_model), "fits": "api", "local": false,
            "measured_tok_s": bench.get(&id).and_then(|b| b["tok_s"].as_f64()), "params_b": Value::Null, "size_gb": Value::Null, "est_tok_s": Value::Null}));
    }
    (servers, models)
}

/// Quality prior (bigger and tool-friendly is better), and speed (measured over estimated).
fn quality(m: &Value) -> f64 {
    let p = m["params_b"].as_f64().unwrap_or(7.0);
    (1.0 + p).ln() * family_bonus(m["name"].as_str().unwrap_or(""))
}
fn speed(m: &Value) -> f64 {
    m["measured_tok_s"].as_f64().or_else(|| m["est_tok_s"].as_f64()).unwrap_or(0.0)
}

fn pick(models: &[&Value], min_speed: f64, exclude: &[String]) -> Option<Value> {
    let ok: Vec<&&Value> = models.iter().filter(|m| !exclude.contains(&m["id"].as_str().unwrap_or("").to_string())).collect();
    let fast: Vec<&&Value> = ok.iter().filter(|m| speed(m) >= min_speed || speed(m) == 0.0).cloned().collect();
    let pool = if fast.is_empty() { ok } else { fast };
    pool.into_iter().max_by(|a, b| quality(a).partial_cmp(&quality(b)).unwrap_or(std::cmp::Ordering::Equal)).map(|m| (*m).clone())
}

fn recommend(hw: &Value, models: &[Value]) -> Value {
    let chat_local: Vec<&Value> = models.iter().filter(|m| m["local"] == true && m["kind"] == "chat" && (m["fits"] == "fits" || m["fits"] == "unknown")).collect();
    let api: Vec<&Value> = models.iter().filter(|m| m["local"] == false && m["kind"] != "embed").collect();
    let embed = models.iter().find(|m| m["kind"] == "embed" && m["local"] == true).map(|m| m["id"].clone());
    let mut roles = serde_json::Map::new();
    let mut why = vec![];
    let solver = pick(&chat_local, 10.0, &[]);
    let improver = pick(&chat_local, 5.0, &[]);
    let id = |m: &Option<Value>| m.as_ref().map(|m| m["id"].as_str().unwrap_or("").to_string());
    let backup = |first: &Option<Value>| pick(&chat_local, 5.0, &id(first).into_iter().collect::<Vec<_>>()).map(|m| m["id"].as_str().unwrap_or("").to_string());
    let chain = |first: &Option<Value>, extra: Option<String>| [id(first), backup(first), extra].into_iter().flatten().collect::<Vec<_>>().join(", ");
    let api_best = api.first().map(|m| m["id"].as_str().unwrap_or("").to_string());
    if let Some(s) = &solver {
        why.push(format!("Solver {}: the strongest local model that should still run at ≥10 tokens/s on this machine.", s["id"].as_str().unwrap_or("")));
    }
    if let Some(i) = &improver {
        why.push(format!("Improver {}: reads long traces, so quality matters more than speed (≥5 tokens/s).", i["id"].as_str().unwrap_or("")));
        if i["params_b"].as_f64().map(|p| p < 7.0).unwrap_or(false) {
            why.push("Your best local model is under 7B parameters: expect weak harness edits. An API improver, or a 14B+ model, helps most.".into());
        }
    }
    let mut presets = vec![];
    if solver.is_some() {
        let s = chain(&solver, None);
        let i = chain(&improver, None);
        presets.push(json!({"id": "all_local", "label": "All local", "about": "free and private; speed depends on this machine", "solver": s, "improver": i, "selector": i}));
        if let Some(a) = &api_best {
            presets.push(json!({"id": "hybrid", "label": "Local solver + API improver", "about": "cheap rollouts, strong harness edits", "solver": s, "improver": format!("{a}, {}", id(&improver).unwrap_or_default()), "selector": format!("{a}, {}", id(&improver).unwrap_or_default())}));
        }
    }
    if let Some(a) = &api_best {
        let fb = id(&solver).map(|s| format!(", {s}")).unwrap_or_default();
        presets.push(json!({"id": "all_api", "label": "All API", "about": "fastest and strongest; costs per token (set a spending cap)", "solver": format!("{a}{fb}"), "improver": format!("{a}{fb}"), "selector": format!("{a}{fb}")}));
    }
    roles.insert("solver".into(), json!(id(&solver)));
    roles.insert("improver".into(), json!(id(&improver)));
    roles.insert("selector".into(), json!(id(&improver)));
    roles.insert("embedding".into(), embed.clone().unwrap_or(Value::Null));
    // What to download: the largest well-behaved model that fits the budget.
    let budget = hw["budget_gb"].as_f64().unwrap_or(0.0);
    let catalog = [("qwen3:4b", 2.6, 4.0), ("qwen3:8b", 5.2, 8.0), ("qwen3:14b", 9.3, 14.0), ("gpt-oss:20b", 13.0, 20.0), ("qwen3:32b", 20.0, 32.0), ("llama3.3:70b", 43.0, 70.0)];
    let have = |n: &str| models.iter().any(|m| m["name"].as_str().map(|x| x == n || x.starts_with(&format!("{n}-"))).unwrap_or(false));
    let mut pulls = vec![];
    if let Some((n, sz, p)) = catalog.iter().rev().find(|(_, sz, _)| sz * 1.15 + 1.5 <= budget) {
        if !have(n) {
            pulls.push(json!({"model": n, "size_gb": sz, "params_b": p, "why": format!("the largest recommended model that fits your {:.0} GB budget", budget)}));
        }
        if let Some((n2, sz2, p2)) = catalog.iter().rev().find(|(m, sz2, _)| *sz2 * 2.2 + 1.5 <= budget || (*sz2 <= sz * 0.5 && m != n)) {
            if !have(n2) && n2 != n {
                pulls.push(json!({"model": n2, "size_gb": sz2, "params_b": p2, "why": "a faster second model for the solver and as a failover link"}));
            }
        }
    }
    if embed.is_none() {
        pulls.push(json!({"model": "nomic-embed-text", "size_gb": 0.3, "params_b": 0.14, "why": "embedding model for semantic memory recall"}));
    }
    json!({"roles": roles, "presets": presets, "why": why, "pulls": pulls})
}

/// GET /api/advisor
pub(crate) async fn advisor() -> ApiResult {
    blocking(move || {
        let hw = hardware();
        let (servers, mut models) = inventory(&hw);
        let rec = recommend(&hw, &models);
        models.sort_by(|a, b| (a["local"] != true).cmp(&(b["local"] != true)).then(quality(b).partial_cmp(&quality(a)).unwrap_or(std::cmp::Ordering::Equal)));
        let n_local = models.iter().filter(|m| m["local"] == true).count();
        Ok(json!({"hardware": hw, "servers": servers, "models": models, "recommend": rec,
            "counts": {"servers": servers.len(), "local_servers": servers.iter().filter(|s| s["kind"] == "local").count(), "custom": servers.iter().filter(|s| s["saved"] == true && s["kind"] == "local").count(), "api": servers.iter().filter(|s| s["kind"] == "api").count(), "local_models": n_local, "fit": models.iter().filter(|m| m["fits"] == "fits").count()}}))
    })
    .await
}

#[derive(Deserialize)]
pub(crate) struct BenchReq {
    id: String,
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    provider: String,
}

/// POST /api/advisor/bench — time a short generation and record tokens/s.
pub(crate) async fn bench(Json(b): Json<BenchReq>) -> ApiResult {
    blocking(move || {
        let (mut conn, model) = match connections::resolve(&b.id) {
            Ok(x) => x,
            Err(_) => {
                // A discovered server not saved yet: bench it directly.
                let (name, model) = b.id.split_once('/').ok_or_else(|| anyhow::anyhow!("model id is connection/model"))?;
                let mut c = connections::resolve("ollama/x")?.0;
                c.name = name.into();
                c.provider = if b.provider.is_empty() { "openai".into() } else { b.provider.clone() };
                c.base_url = b.base_url.clone();
                (c, model.to_string())
            }
        };
        if !b.base_url.is_empty() && conn.base_url.is_empty() {
            conn.base_url = b.base_url.clone();
        }
        let msgs = vec![client::Msg::User("Count from 1 to 60, separated by spaces. Output only the numbers.".into())];
        let t0 = Instant::now();
        let r = client::chat(&conn, &model, &client::ChatReq { system: "", messages: &msgs, tools: &[], max_tokens: 200 }, Duration::from_secs(300))?;
        let secs = t0.elapsed().as_secs_f64();
        let out_tok = r.output_tokens.max(1) as f64;
        let res = json!({"id": b.id, "seconds": round1(secs), "out_tokens": r.output_tokens, "tok_s": round1(out_tok / secs.max(0.01)), "at": util::now_iso()});
        let mut g = BENCH.lock().unwrap();
        g.insert(b.id.clone(), res.clone());
        let _ = util::write(&crate::config::get().runs.join("bench.json"), serde_json::to_string_pretty(&*g)?);
        Ok(res)
    })
    .await
}

#[derive(Deserialize)]
pub(crate) struct PullReq {
    model: String,
    #[serde(default)]
    base_url: String,
}

static PULLS: Lazy<Mutex<BTreeMap<String, String>>> = Lazy::new(|| Mutex::new(BTreeMap::new()));

/// POST /api/advisor/pull — ask Ollama to download a model (in the background).
pub(crate) async fn pull(Json(p): Json<PullReq>) -> ApiResult {
    let base = if p.base_url.is_empty() { "http://127.0.0.1:11434".to_string() } else { p.base_url.trim_end_matches('/').trim_end_matches("/v1").to_string() };
    if !Regex::new(r"^[a-zA-Z0-9._:/-]+$").unwrap().is_match(&p.model) {
        return Err(anyhow::anyhow!("not a model name").into());
    }
    PULLS.lock().unwrap().insert(p.model.clone(), "downloading…".into());
    let model = p.model.clone();
    std::thread::spawn(move || {
        let r: Result<()> = (|| {
            ureq::AgentBuilder::new().timeout_read(Duration::from_secs(6 * 3600)).try_proxy_from_env(false).build().post(&format!("{base}/api/pull")).send_json(json!({"name": model, "stream": false}))?;
            Ok(())
        })();
        PULLS.lock().unwrap().insert(model.clone(), match r { Ok(()) => "done".into(), Err(e) => format!("failed: {e}") });
    });
    Ok(Json(json!({"started": p.model})))
}

pub(crate) async fn pull_status() -> ApiResult {
    Ok(Json(json!(PULLS.lock().unwrap().clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes_and_ranks_sensibly() {
        assert_eq!(parse_params("qwen2.5:7b-instruct-q4_K_M"), Some(7.0));
        assert_eq!(parse_params("7.6B"), Some(7.6));
        assert_eq!(parse_params("Llama-3.2-1B-Instruct"), Some(1.0));
        assert_eq!(kind_of("nomic-embed-text:latest"), "embed");
        let hw = json!({"budget_gb": 12.0, "bandwidth_gbs": 200.0});
        let m = |id: &str, p: f64, sz: f64| json!({"id": id, "name": id, "local": true, "kind": "chat", "fits": if sz * 1.15 + 1.0 <= 12.0 { "fits" } else { "too big" }, "params_b": p, "size_gb": sz, "est_tok_s": 200.0 / sz * 0.6});
        let models = vec![m("o/qwen3:8b", 8.0, 5.2), m("o/llama3.2:1b", 1.0, 1.3), m("o/qwen3:32b", 32.0, 20.0)];
        let r = recommend(&hw, &models);
        assert_eq!(r["roles"]["solver"], "o/qwen3:8b", "the 32B does not fit 12 GB; 8B beats 1B");
        assert!(r["presets"].as_array().unwrap().iter().any(|p| p["id"] == "all_local" && p["solver"].as_str().unwrap().contains("o/llama3.2:1b")), "second model is the failover link");
        assert!(r["pulls"].as_array().unwrap().iter().any(|p| p["model"] == "nomic-embed-text"));
    }
}
