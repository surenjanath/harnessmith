# Upstream issues, checked against harnessmith

Every issue on [google-research/harness2](https://github.com/google-research/harness2/issues) as of 2026-11-05 (38 open, 7 closed), checked against this port's code.

Status:
- ✅ **fixed here**: fixed in this port.
- 🟢 **not affected**: we already behave correctly.
- 🔜 **to do**: applies to us, not done yet.
- ⚪ **n/a**: doesn't apply (packaging of the Python repo, questions, off-topic).

## Bugs

| # | Upstream issue | Status | What harnessmith does |
|---|---|---|---|
| [156](https://github.com/google-research/harness2/issues/156) | Quoted YAML skill names (`name: "x"`) rejected | ✅ fixed here | Quotes are stripped before comparing; a different name is still rejected. Test: `quoted_skill_names_match_their_directory`. |
| [153](https://github.com/google-research/harness2/issues/153) | Same-named images from different folders overwrite each other | ✅ fixed here | A repeated basename is prefixed with its parent folder (then a counter). |
| [151](https://github.com/google-research/harness2/issues/151) / [185](https://github.com/google-research/harness2/issues/185) | Windows path separators break harness propagation at k≥2 | 🟢 not affected | Tree keys are always built with `/` (`util::rel`), and locking uses the cross-platform `fs2`. |
| [143](https://github.com/google-research/harness2/issues/143) | Enforcement gaps | ✅ gap 1b · 🔜 rest | 1b: one `is_absent_evidence()` covers every "no evidence" placeholder, in selection and the sweep. Still to do: 1a (ledger tracking live vs dead rollouts), 1c (re-check the budget at judge/report time), 2 (library tombstones for declined or failed lessons), 3 (provenance on lessons). |
| [109](https://github.com/google-research/harness2/issues/109) | Improver outage saved as `ok`; resume never re-runs | 🟢 not affected | `execution_failed` makes the mode not ok, so resume re-runs it. The UI also retries runs that end on transient errors. |
| [98](https://github.com/google-research/harness2/issues/98) | Report drops a mode's improver spend when it fails after its stages ran | ✅ fixed here | The error is added to the mode record instead of replacing it, so the stages and their spend survive. |
| [50](https://github.com/google-research/harness2/issues/50) | Report undercounts spend for tasks re-run from cache | ✅ fixed here | Cache replays carry `replayed_usd` (the original cost) into each stage, and the report counts paid + replayed. |
| [80](https://github.com/google-research/harness2/issues/80) | `judge_one` trusts a null `all_pass` / `n_passed` | 🟢 not affected | `has_grade` requires both numbers, and a null `all_pass` falls back to `n_passed == n_criteria`. |
| [44](https://github.com/google-research/harness2/issues/44) | Judge keeps an old grade after a re-run | 🟢 not affected | Merge keeps an old grade only for the same `run_id` and `selected_at`. |
| [65](https://github.com/google-research/harness2/issues/65) | Caches reuse outputs after task inputs change | 🟢 not affected | Task inputs are part of every rollout's fingerprint (goal, JobBench, LAB). |
| [121](https://github.com/google-research/harness2/issues/121) | Rollout killed at its time limit recorded as finished | 🟢 not affected | The runner records the exit code (124, 137, signals) as an `error` event. |
| [111](https://github.com/google-research/harness2/issues/111) | Rollouts share npm/pip installs, temp files and workspaces | 🟢 mostly | Native solver: the bubblewrap sandbox gives every command its own `/tmp` and a hidden home, with no access to other workspaces. The OpenCode runner follows upstream's fix. |
| [77](https://github.com/google-research/harness2/issues/77) | LAB/OpenCode runner fails silently without GNU `timeout` (macOS) | ✅ fixed here | Uses `timeout` or `gtimeout`, and otherwise fails with a clear message (`brew install coreutils`). |
| [38](https://github.com/google-research/harness2/issues/38) | Judge/report print tracebacks for bad arguments | 🟢 not affected | Every CLI error is one `error:` line. |
| [95](https://github.com/google-research/harness2/issues/95) | LAB judge can't route to OpenAI-compatible endpoints | 🟢 goal mode / ⚪ LAB | Goal-mode judging uses any connection (OpenAI-compatible, Anthropic, Gemini, Ollama). The LAB judge is upstream Harvey code. |
| [146](https://github.com/google-research/harness2/issues/146) / [144](https://github.com/google-research/harness2/issues/144) | `third_party/` missing from wheels / sdist | 🔜 to do (our version) | Our release tarballs ship only the binary. Goal mode and the UI need nothing else, but benchmark mode needs `third_party/`. To do: ship it in the tarball, or document it. |

## Features and proposals

| # | Upstream issue | Status | What harnessmith does |
|---|---|---|---|
| [4](https://github.com/google-research/harness2/issues/4) | `--max-usd` spend limit | ✅ done | A per-run cap (agents stop, run ends `capped`) and daily per-connection caps that fail over along the chain. |
| [5](https://github.com/google-research/harness2/issues/5) | Run on your own folder of tasks | ✅ done | Goal projects: type a goal or attach files in the UI, or `harness2 goal`, `harness2 solve`, the webhook, GitHub issues or schedules. |
| [161](https://github.com/google-research/harness2/issues/161) | Deterministic verification gate inside a k=1 trajectory | ✅ done (general form) | `llm/review.rs` checks deliverables against the goal before `finish` (file types, complete HTML, styling, depth, broken JSON/CSV, error text), with no model call. |
| [148](https://github.com/google-research/harness2/issues/148) / [147](https://github.com/google-research/harness2/issues/147) | Docker container runs as root | ✅ done | The Dockerfile runs as an unprivileged `harness` user. |
| [100](https://github.com/google-research/harness2/issues/100) / [64](https://github.com/google-research/harness2/issues/64) / [117](https://github.com/google-research/harness2/issues/117) | Antigravity, Pi, Claude Code, Hermes substrates | 🟢 partly | The native runtime works with any OpenAI-compatible, Anthropic, Gemini or Ollama model, and the improver can run through Claude Code (`claude_cli`). Dedicated Pi or Hermes substrates are not planned. |
| [94](https://github.com/google-research/harness2/issues/94) | Pin harness components the improver can't remove | 🔜 to do | Plan: a project setting `pinned_paths`, outside the improver's view. Proposals that change pinned files are rejected and retried, and pinned content is checked by hash before each rollout. (The operator gate already lives outside the harness.) |
| [96](https://github.com/google-research/harness2/issues/96) | Static check on an evolved harness | 🔜 to do | Plan: goal-mode validation rejects hidden or bidirectional Unicode characters in harness files, and Doctor scans base harnesses. |
| [2](https://github.com/google-research/harness2/issues/2) | Opt-in human feedback for selection and lessons | 🔜 to do | Plan: "Prefer this output" on the compare tab records a human selection, re-runs reflection, and tags the lesson `human`, listed first. |
| [149](https://github.com/google-research/harness2/issues/149) | Use in production without Propose–Probe–Compose on every task | 🟢 partly · 🔜 versioning | Already: "Use as project base harness", memory promotion into `LongTermMEMORY.md`, A/B tests and shootouts, schedules. To do: base-harness version history with rollback. |
| [106](https://github.com/google-research/harness2/issues/106) | Publish a version / tag | 🔜 to do (ours) | Tag `v0.1.0`; the release workflow then builds binaries and a Docker image. |
| [57](https://github.com/google-research/harness2/issues/57) / [61](https://github.com/google-research/harness2/issues/61) / [55](https://github.com/google-research/harness2/issues/55) / [59](https://github.com/google-research/harness2/issues/59) | Outcome-based getting started; when to use it; credential-free example | 🔜 to do (docs) | Plan: `docs/GETTING_STARTED.md`, covering a first goal in 5 minutes with the offline mock model (no keys), then a local model, when Harness² helps, and when a single run is enough. |
| [179](https://github.com/google-research/harness2/issues/179) | Benchmark × executor catalog | ⚪ n/a | Upstream refactor (closed). |
| [136](https://github.com/google-research/harness2/issues/136) | Native subscription workflow for LAB | ⚪ n/a | Specific to upstream's LAB setup. |
| [178](https://github.com/google-research/harness2/issues/178), [91](https://github.com/google-research/harness2/issues/91), [71](https://github.com/google-research/harness2/issues/71), [39](https://github.com/google-research/harness2/issues/39) | Skill conversion, LAB study, analysis write-up, citation request | ⚪ n/a | Discussion, not code. |
| [73](https://github.com/google-research/harness2/issues/73), [152](https://github.com/google-research/harness2/issues/152), [155](https://github.com/google-research/harness2/issues/155) | Off-topic / spam | ⚪ n/a | |
| [194](https://github.com/google-research/harness2/issues/194) | Our announcement | — | |
