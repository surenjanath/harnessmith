# Harness² (Rust)

A Rust port of [google-research/harness2](https://github.com/google-research/harness2), *Recursive Agent Harnessing for an Open World*, plus a model-agnostic agent runtime and a web UI where you connect models and hand the system a goal.

A **harness** is everything around a model that makes it an agent: system prompt, guardrails, memory, tool guidance, skills, sub-agents, scripts, and in-loop hooks or plugins. Harness² improves that harness for each task, with no ground-truth feedback:

1. **Probe the base.** The solver attempts the task k times with the current harness.
2. **Propose.** An improver model reads those trajectories and proposes edits along two dimensions: how the agent *works* (INV) and what a *practitioner knows* (DOM).
3. **Probe the edits.** Each candidate harness gets its own fresh attempt.
4. **Compose.** A composer merges what the evidence supports. In *parallel* mode, k independent compositions; in *sequential* mode, k rounds that build on each other.
5. **Select blind.** A selector picks the best final output without knowing which harness produced it.
6. **Learn.** A reflection writes a lesson into a per-domain library, which later tasks read.

Budget per task and mode: R = 4k solver rollouts and P = 3k improver sessions. The ledger enforces this.

## Quick start

```bash
cargo build --release
```

```bash
./target/release/harness2 demo --k 2
```

The demo runs both recursion modes with scripted editors and real Python executions. It needs no credentials, only `python3`.

```bash
./target/release/harness2 serve
```

Then open <http://127.0.0.1:8765>:

- **Models**: add a connection (OpenAI-compatible, Anthropic, Gemini, Ollama), fetch its model list, test it, and choose models for the solver, improver and selector roles.
- **Run a goal**: type the task, attach input files, pick parallel or sequential mode and the budget k, then watch the live log.
- **History**: the adopted output, why it was picked, the harness diff against base, every stage's notes, and the trajectory.
- **Base harness**: edit the starting harness of a project. **Library**: the lessons learned so far.

Goals in the same project share a library. Give related goals one project and one domain so later goals benefit from earlier ones.

## Connecting models

A model id is `connection/model`. These prefixes work with no setup if the key is in your environment:

| prefix | provider | key variable |
|---|---|---|
| `openai/…` | OpenAI | `OPENAI_API_KEY` |
| `anthropic/…` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini/…`, `google/…` | Google Gemini | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `ollama/…` | local Ollama | none |
| `openrouter/`, `groq/`, `deepseek/`, `together/`, `mistral/` | OpenAI-compatible | `<NAME>_API_KEY` |
| `mock/…` | deterministic offline stand-in for smoke tests | none |

You can also register any other OpenAI-compatible server (vLLM, LM Studio, a gateway) in the UI or from the command line:

```bash
harness2 models add local --provider openai --base-url http://localhost:8000/v1 --default-model qwen3-32b
```

```bash
harness2 models test local/qwen3-32b
```

```bash
harness2 models role solver local/qwen3-32b
```

Connections are saved to `connections.json` (override with `HARNESS2_CONNECTIONS`). Prefer `api_key_env` over a literal key; the file is written with mode 0600. Loopback and `NO_PROXY` hosts bypass `HTTPS_PROXY`.

## Watching a run: the pipeline map

While a goal runs, **Run a goal** draws its pipeline live, one lane per recursion mode:

```text
parallel    base runs ─► INV/DOM candidates (edit + probe) ─► compositions (edit + probe) ─► select (blind) ─► learn
sequential  base runs ─► round 1 inv/dom ─► checkpoint c1 ─► round 2 inv/dom ─► c2 … ─► select ─► learn
```

Each card shows its state as it changes: which tool the agent is calling, step count, tokens, whether the improver *kept* or *edited* the harness (or fell back after an invalid proposal), and whether the probe delivered. The adopted output is marked ★. Click a card for its timeline: every agent turn, the proposal notes, rejected attempts and why, and the selector's reasoning. The header tracks elapsed time, solver runs and improver sessions against the 4k/3k budget, tokens, and cost. **Stop** cancels after the current model call.

Runs are saved under `runs/ui-runs/<id>/` (`run.json`, `log.txt`, `events.jsonl`), so the map and log survive a server restart. **History → Journey** replays the map for any past goal.

## Local models

The UI's **Detect local servers** probes Ollama (11434), LM Studio (1234), llama.cpp (8080/8081), vLLM (8000), text-generation-webui (5000) and LocalAI (8088), and adds what it finds. Or add one by hand:

```bash
harness2 models add ollama --provider ollama --default-model qwen2.5:7b
```

```bash
harness2 models add lms --provider openai --base-url http://localhost:1234/v1 --default-model qwen2.5-7b-instruct
```

What the runtime does for small local models:

- **Ollama uses its native `/api/chat`**, which honours `num_ctx`. Ollama's `/v1` shim silently cuts prompts to a 2-4k window, and a Harness² mission alone is longer than that. Set **Context window** on the connection (default 32768 for Ollama).
- **Tool calls written as text are recovered.** `<tool_call>{…}</tool_call>` (Qwen/Hermes), `<|python_tag|>{…}` (Llama), and fenced or bare JSON with `name` + `arguments` become real tool calls. A reply that looks like a call but doesn't parse gets a retry prompt instead of ending the session.
- `<think>` blocks are stripped, empty replies get a nudge to continue, and old observations are trimmed to fit the context window.
- **Max parallel requests** per connection (default 1 for Ollama) queues requests instead of timing them out. With one local model serving every role, also set **Parallel stages = 1** (Models → Advanced) so the server's prompt cache is reused across an agent's turns. Otherwise interleaved sessions re-evaluate the whole prompt every turn.
- A rollout whose session died before producing anything (server error, cancel) is reported as failed and is not cached, so a rerun tries it again.

## Goals from the command line

```bash
harness2 goal "Find the three regions with the largest decline in sales.csv and write output/brief.md" \
  --file sales.csv --solver openai/gpt-4.1 --improver anthropic/claude-sonnet-4-5 --k 2 --mode parallel
```

Rollouts run on the **native substrate** (`nx`): this crate's own tool loop with `list_files`, `read_file`, `write_file`, `edit_file`, `delete_file`, `bash`, `skill`, `task` (sub-agents) and `finish`. File tools are sandboxed to the workspace, and `hooks.json` hooks (`PreToolUse`, `PostToolUse`, `Stop`) run inside the loop. Turn the shell tool off in project settings or with `HARNESS2_ALLOW_SHELL=0`. The improver and selector run on the native `api` channel by default; `opencode` and `claude_cli` are also available.

To check a model on a task before paying for a full run, `harness2 solve` makes one solver attempt on the base harness and prints the output, steps, and tokens:

```bash
harness2 solve "Summarise sales.csv in output/brief.md" --file sales.csv --model ollama/qwen2.5:7b
```

An optional rubric (one criterion per line) is stored with the goal and never shown to any model. The **Judge** button, or `harness2 judge --bench goal --subset <project>`, grades the adopted output against it with the judge model.

## The benchmarks

The reference CLIs are subcommands, and the flags match the original:

| Python | Rust |
|---|---|
| `harness2` | `harness2 run` |
| `harness2-judge` | `harness2 judge` |
| `harness2-report` | `harness2 report` |
| `harness2-setup` | `harness2 setup` |
| `harness2-check` | `harness2 check` |
| `harness2-demo` | `harness2 demo` |
| `harness2-bridge` | `harness2 bridge` |

```bash
cp .env.example .env
```

```bash
harness2 setup lab
```

```bash
harness2 check --bench lab
```

```bash
harness2 run --bench lab --k 1 --mode parallel --only-domains antitrust-competition --max-tasks 1
```

```bash
harness2 judge --bench lab --k 1 --mode parallel
```

```bash
harness2 report --bench lab --k 1 --mode parallel
```

JobBench (`jb`), LAB (`lab`) and WorkBuddy-Bench (`wb --subset office|web|code`) run on OpenCode (`--substrate oc`) or Codex (`--substrate cx`), exactly as upstream. Their datasets, graders and Harbor runtime are Python projects that `setup` installs at the pinned commits. This port drives them in subprocesses, as the original does. `third_party/` holds the upstream runner overlays, patches and licenses. See the upstream [setup](https://github.com/google-research/harness2/blob/main/docs/SETUP.md) and [benchmark](https://github.com/google-research/harness2/blob/main/docs/BENCHMARKS.md) guides for prerequisites (Node 20+, uv, gcloud, Docker for WorkBuddy, Pandoc/Poppler for LAB). You can also use `--bench goal --subset <project>` with `run`, `judge` and `report`.

## Fidelity to the reference implementation

- Every method prompt is byte-identical to the Python output. A golden test covers 87 prompt variants across both prompt packs and both surfaces.
- Seeded stream order and anonymous candidate ids are bit-exact with CPython's `random.Random`, through a Mersenne Twister port, so namespaces, caches and resumes match.
- Harness fingerprints, input fingerprints, run markers, summary merging and resume rules, the leak and grader-verdict tripwires, the validation messages fed back to proposers, and the budget ledger all follow the original.
- The demo reproduces the reference counts: 28 executions and 32 editor calls at k=2, with R=8 and P=6 per mode.

## Layout

```text
src/core/        spec, tree, prompts, context, proposer, recursion, selection, schedule,
                 library, ledger, runmark, judge, report, pyrandom
src/substrate/   opencode, codex, native
src/llm/         connections, client (OpenAI / Anthropic / Gemini / Ollama / mock), agent loop
src/benchmarks/  demo, goal, jobbench (+ jb_oc_runner), lab, workbuddy (+ wb_render)
src/cli/         subcommands
src/ui/          axum server + embedded single-page UI
```

```bash
cargo test
```

## Notes

- The UI has no authentication and the agent can run shell commands. Keep it on 127.0.0.1.
- What has been verified: the demo, the goal pipeline on the mock model, a strict OpenAI-compatible test server, a real local model (Qwen2.5-3B through llama.cpp's server, end to end through the UI), and the UI in a headless browser. See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for results and the roadmap. The Anthropic and Gemini clients follow their public APIs but have not been run against live keys here. The JobBench, LAB and WorkBuddy adapters compile and are unit-tested, but have not been run against the installed benchmark datasets.

## License

Apache 2.0. Based on Harness² by Google LLC (not an officially supported Google product). Third-party material keeps its upstream licenses; see `third_party/` and `dependencies/`.

```bibtex
@article{xu2026harness2,
  title   = {Harness$^2$: Recursive Agent Harnessing for an Open World},
  author  = {Xu, Ruiyao and Chen, Yanfei and CuiZhu, Zhongying and Dalvi Mishra, Bhavana and Ming, Yifei and Yu, Han and Han, Rujun and Lee, Chen-Yu and Pfister, Tomas},
  journal = {arXiv preprint arXiv:2610.14224},
  year    = {2026}
}
```
