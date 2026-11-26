# Testing notes and roadmap

## What was tested

| Area | How |
|---|---|
| Method fidelity | 87 prompt variants byte-identical to the Python reference; CPython-exact seeded order and anonymous ids; harness/input fingerprints |
| Whole method | the deterministic demo (both modes, k=2): 28 executions, 32 editor calls, R=8/P=6 per mode, library threading, resume |
| Model wire formats | a strict OpenAI-compatible test server (string content, `tool` messages, a text-emitted `<tool_call>` round trip) |
| Local models | llama.cpp's server (llama-cpp-python 0.3.16) with Qwen2.5-3B-Instruct Q4_K_M on 4 CPU cores, through the UI, end to end |
| UI | headless Chromium: connections, roles, goal submission with files, the live pipeline map, node timelines, History journey, mobile width |

## What the local-model run found and fixed

1. **Null assistant content.** OpenAI accepts `content: null` beside `tool_calls`; llama.cpp and other strict servers return 500. The client now sends `""`.
2. **Text tool calls.** Qwen2.5 through a generic template writes `<tool_call>{…}</tool_call>` as text. These are now parsed into real calls, and a malformed one gets a retry prompt instead of ending the session as a "final answer".
3. **Dead rollouts were cached.** A rollout whose session died before producing anything was marked complete and reused on rerun. It is now reported as failed and left unmarked, as the reference requires.
4. **Prompt-cache thrash.** Two stages in parallel against one local server interleave sessions, so every turn re-evaluates the whole prompt (~29 tok/s here). The new *Parallel stages = 1* setting keeps each session's prefix cached, and turns drop to ~10 s.
5. **Context truncation.** Ollama's `/v1` shim cuts prompts to its default window. Ollama now goes through `/api/chat` with `num_ctx`.
6. **Malformed JSON from small models.** Qwen-3B writes tool arguments with raw newlines and invalid escapes (`\'`) inside JSON strings, and repeating the request doesn't fix it (it failed the same way three times in a row). A lenient parser now repairs these, and the exact failing reply is a regression test (`tests/golden/qwen_broken_call.txt`).
7. **Unhelpful tool errors.** The model asked for `input/regions.csv` while the file was at the workspace root, and the bare error didn't help it recover. Errors now name the closest file (“Did you mean `regions.csv`?”), list the workspace, spell out missing required arguments, and use relative paths.
8. **Discovery on a busy server.** A local server busy generating can't list its models in time. It is now reported with a note instead of hidden.

**Solver-only trials, same task (correct answer: North, −25.0%).** Before fixes 5–7: 0 of 6 attempts produced the answer (placeholder tables, invented data, or a raw tool call dumped as the output). After: 1 of 3 named North at −25.0% with a full table (three other rows had arithmetic slips), and 2 of 3 ran out of the 20-step budget. The sample is too small to claim a rate, but the failures moved from runtime problems to the model's own reasoning limits, which is what a runtime fix can deliver.

**Result on Qwen2.5-3B (k=1, parallel, 27.5 min on CPU).** Every stage ran and the map tracked it. But the 3B improver kept the base harness in all three edit sessions: it cannot follow a 6–8k-token improvement mission. The adopted output named the right region with a wrong percentage. **The method needs a capable improver.** Use a small local solver if you like, but give the improver and selector a stronger model (local 30B+ or an API model). Roles are configured separately, so a local solver with an API improver works out of the box.

## Chaos test: self-healing on local models

Setup: two llama.cpp servers (Qwen2.5-3B on :8081, Llama-3.2-1B on :8082). Every role was set to `local/qwen2.5-3b, llama/llama3.2-1b`.

1. **Kill the primary model mid-rollout.** The Qwen server was killed 54 s into base1. The agent failed over to Llama at once and base1 still delivered. Before the fix, failover took 28 s, because connection-refused errors were retried 4 times. They now fail fast.
2. **Kill the UI server mid-run.** On restart the server logged `auto-resuming interrupted run … as …`. The resumed run reused the completed base1 from cache. This was done twice, and the second restart was the upgrade to a new binary.
3. **Bring the primary back.** inv1 failed over to Llama while Qwen was down. It switched back 35 s later (`recovered: back from llama/llama3.2-1b to local/qwen2.5-3b`). The map showed "Self-healing ×2 — model failover ×1 · recovered ×1".

Found and fixed along the way:
- failover events were reported twice, and a session could bounce back to a dead model. A per-session avoid list fixed both;
- the time estimate read ~771 min, because tokens/s on local servers includes prompt time. It now uses measured seconds per call × expected steps (~18 min for the same goal);
- Doctor marked a busy local server, one that is generating and doesn't answer the probe, as unreachable. It now counts recent successful calls.

The compare tab also showed something about the method. On an earlier 3B run, the adopted harness (agg1) wrote "No regions.csv found", while the unmodified base1 had produced a table. Blind selection with a single final candidate can adopt a regression. See roadmap item 12.

## Real-model trials through Codex (ChatGPT sign-in)

The `codex` provider runs every role on a capable hosted model with no API key: each model call is one stateless `codex exec` (Codex 0.165, `gpt-6.1-terra`). The harness keeps its own tools; Codex only answers with JSON (`--output-schema`) holding the reply and the tool calls.

**Token cost per call.** Codex's own coding-agent prompt, tool schemas and skills list were ~17.5k input tokens on every call. Replacing its instructions (`model_instructions_file`), switching off its tools and context blocks, and running in one fixed folder (a random folder per call broke the prompt-cache prefix) brought that to ~4.3k. On the same 4-step solver task: 63.7k → 13.9k uncached input tokens (−78%), same answer.

**Hard task** (messy multi-file data: three date formats, three currencies with monthly rates, 18 duplicate orders, refunds and cancellations, a join to regions; ground truth computed independently). Full goal, k=1, Codex in every role, from the web UI:

| | time | tokens | base1 | adopted output |
|---|---|---|---|---|
| before the fixes below | 6m28s | 532k | 11 steps, 1m25s | every figure right, padded to 785 words, flagged "needs review" |
| after | 4m29s | 354k | 6 steps, 47s | every figure right, 178 words, clean |

Found and fixed on the way:
1. **An input read as a requested output.** "convert each order with the fx_rates.csv rate" made a CSV a required deliverable, so every attempt's finish was refused (×11) and the run ended "needs review". Attached inputs, and names after "using/with/from/in/of", no longer count.
2. **Essays demanded for tables.** "report" in the file name `output/report.md` triggered the 600-word depth check, so a correct table was refused until it was padded. File names no longer count, and a computation (table, totals, top N) needs depth only when the goal asks for it.
3. **A stalled `codex exec`** hung one call for 13 minutes, because the agent passes its whole remaining time as the call timeout. Each call is now capped (`HARNESS2_CODEX_CALL_TIMEOUT`, default 600 s), and a killed call is retried or fails over.
4. **Trailing junk after a tool call's JSON arguments** (`{…}"}]}=} CX1;…`) lost the call. The first complete object is now used.
5. **Observation masking.** Old tool outputs are re-sent on every turn. Once stale outputs (older than the last 6 messages, over 4k chars each) add up past 24k chars, they are cut to a 1.5k head together, in one batch so the cached prefix survives between batches (`HARNESS2_OBS_MASK=0` turns it off). Studies of coding and tool-use agents report this halves cost without lowering solve rates.
6. The UI said "refused finish with nothing written" for any refused finish. It now names the review's reason.

`HARNESS2_CODEX_DUMP=<dir>` keeps every prompt and event stream, to see what a call cost.

## Learning test, economy and local models

**Learning with a strong solver.** Three related hard data goals in one project, then the third again in a fresh project as a control, all with Codex in every role. The data had traps the goals never mention (European amounts like `€1.234,50`, day-first dates, stale duplicate rows, refunds converted at the original order's month). Every attempt got every figure right (64/64), all 12 improver stages kept the base harness, and the library got only "no lesson" records: with nothing failing, there is nothing for harness edits to fix. Goal 3 after goals 1–2 took 4m58s, the control 5m16s.

What changed because of it:
1. **Unchanged harnesses are not run again** (goal mode). A stage whose harness is byte-identical to the base reuses the base attempt; it still counts in the ledger. Candidates that are the same run, or delivered the same text, are one candidate, so no selector session compares an output with itself.
2. **No reflection without edits.** When every stage kept the base, the Learn step records "no edits" instead of spending a session to find nothing.
3. **Automatic data notes.** Strong models rarely call `remember` (one note in four runs). After a successful session that saved none, the solver's model is asked once for up to three durable facts about the data, kept as candidate notes that become active if the attempt is adopted. First live result: "EUR uses periods for thousands and commas for decimals; USD and GBP use the reverse", among others. `HARNESS2_AUTO_NOTES=0` turns it off.
4. **Terminal runs show in the UI.** `harness2 goal` marks its project while it runs; History shows "running (terminal)" and the header shows a chip.

Benchmark mode keeps the paper's accounting: reuse and the skipped reflection are off there.

**Local models on an M1 Max (64 GB), same hard task as a solver attempt.**

| model | generation | result |
|---|---|---|
| gpt-oss:20b (MoE) | 61 tok/s | 2m46s, wrong: read `€1.234,50` as 123450 (total 3.6× too high) |
| gemma4:26b-mlx (MoE) | 49 tok/s, ~12 tok/s prompt | stopped after 18 min and 3 tool calls |
| qwen3.6:35b-a3b-mxfp8 (MoE) | 51 tok/s, ~490 tok/s prompt | rerun pending after the fix below |
| qwen3.8:27b-mlx (dense) | 17.6 tok/s | too slow for agent loops |

**Thinking models returned empty replies on Ollama.** Qwen3.x and Gemma 4 think by default there; a long hidden reasoning used the whole reply budget (0 characters of answer after 1,019 of thinking), so the accuracy boost's first strategy "gave nothing" on every attempt. The Ollama client now sends `think: false` unless a connection sets `thinking` ("on" or a level), and retries without the flag for models that refuse it.

**Ollama serializes parallel agents.** On gpt-oss:20b, 4 parallel requests got 42.1 tok/s in total against 42.0 for one: the harness's parallel stages run one after another on a local model. A separate project, kiln, is a local model server built for agent workloads (parallel slots, per-agent prompt cache, speculative decoding) behind Ollama- and OpenAI-compatible APIs.

Accuracy cross-checks for strong models stay opt-in (Accuracy boost "On"): Codex was right on 16 of 16 attempts, so voting would double the cost for no measured gain.

## Roadmap, by expected value

1. **Hybrid presets.** One-click role presets ("local solver + API improver", "all local", "all API"). The cost and time estimate before Start is now built.
2. **Provider prompt caching.** Improver missions repeat large evidence trees across attempts and stages. Anthropic `cache_control`, Gemini context caching and OpenAI automatic caching would cut improver cost substantially.
3. **A compact prompt pack for small improvers.** An optional third pack (beside `adapt` and `tts`) with shorter missions and a pre-digested evidence summary. It departs from the paper, so it should be off by default.
4. **Stronger solver sandbox.** File tools are confined to the workspace, but `bash` is not, and root ignores file permissions. Run solver commands in bubblewrap/nsjail or a container, mount inputs read-only, and cut network access during rollouts (the reference warns that benchmark answers are online).
5. **Push updates.** The UI polls every 1.2 s. Server-sent events would make the map instant and cheaper.
6. **Mid-task resume.** Resume is per task: a stopped run re-rolls unfinished stages, and only the content-addressed proposer cache and completed rollouts are reused. Persisting stage results would resume exactly where it stopped.
7. **Per-connection budgets.** The run queue is built (`HARNESS2_MAX_RUNS`). Spend caps per connection would stop a runaway API bill.
8. **Library retrieval across domains.** The paper keeps libraries strictly per domain. An opt-in, embedding-ranked "related domains" view would help sparse new domains.
9. **Rubric-guided reporting.** Goals with a rubric could show judged base vs adopted scores side by side, to show whether adaptation helped on your own tasks.
10. **Benchmark parity runs.** Run JobBench, LAB and WorkBuddy against the installed datasets and compare a few tasks' summaries with the Python implementation's.
11. **UI authentication.** The server is for localhost; serving it more widely needs a login and per-user projects.
12. **Guard against adopting a regression.** Keep base1 as a candidate in selection when the adapted output is empty or says the inputs are missing, or flag it in the UI.
13. **A persistent Codex process.** Each call starts `codex exec` (~1–2 s of a ~4–5 s call). The Codex app-server keeps one process and takes client-side tools on `thread/start`, which would also give native tool calls instead of JSON in the prompt.
14. **Health-aware chains.** Order a chain by measured success rate and latency instead of fixed order, and share breaker state across server restarts.
