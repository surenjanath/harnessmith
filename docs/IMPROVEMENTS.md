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
6. **Discovery on a busy server.** A local server busy generating can't list its models in time. It is now reported with a note instead of hidden.

**Result on Qwen2.5-3B (k=1, parallel, 27.5 min on CPU).** Every stage ran and the map tracked it. But the 3B improver kept the base harness in all three edit sessions: it cannot follow a 6–8k-token improvement mission. The adopted output named the right region with a wrong percentage. **The method needs a capable improver.** Use a small local solver if you like, but give the improver and selector a stronger model (local 30B+ or an API model). Roles are configured separately, so a local solver with an API improver works out of the box.

## Roadmap, by expected value

1. **Hybrid presets.** One-click role presets ("local solver + API improver", "all local", "all API") with a cost and time estimate before Start, from connection prices and measured tokens/s.
2. **Provider prompt caching.** Improver missions repeat large evidence trees across attempts and stages. Anthropic `cache_control`, Gemini context caching and OpenAI automatic caching would cut improver cost substantially.
3. **A compact prompt pack for small improvers.** An optional third pack (beside `adapt` and `tts`) with shorter missions and a pre-digested evidence summary. It departs from the paper, so it should be off by default.
4. **Stronger solver sandbox.** File tools are confined to the workspace, but `bash` is not, and root ignores file permissions. Run solver commands in bubblewrap/nsjail or a container, mount inputs read-only, and cut network access during rollouts (the reference warns that benchmark answers are online).
5. **Push updates.** The UI polls every 1.2 s. Server-sent events would make the map instant and cheaper.
6. **Mid-task resume.** Resume is per task: a stopped run re-rolls unfinished stages, and only the content-addressed proposer cache and completed rollouts are reused. Persisting stage results would resume exactly where it stopped.
7. **Run queue.** Goals currently run concurrently with no limit. A queue with per-connection budgets would let a single local GPU serve several goals in turn.
8. **Library retrieval across domains.** The paper keeps libraries strictly per domain. An opt-in, embedding-ranked "related domains" view would help sparse new domains.
9. **Rubric-guided reporting.** Goals with a rubric could show judged base vs adopted scores side by side, to show whether adaptation helped on your own tasks.
10. **Benchmark parity runs.** Run JobBench, LAB and WorkBuddy against the installed datasets and compare a few tasks' summaries with the Python implementation's.
11. **UI authentication.** The server is for localhost; serving it more widely needs a login and per-user projects.
