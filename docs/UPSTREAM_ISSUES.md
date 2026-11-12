# Upstream issues, checked against harnessmith

Every issue on [google-research/harness2](https://github.com/google-research/harness2/issues) as of 2026-11-05 (38 open, 7 closed), checked against this port's code.

**Still to do:** #106 (tag a release), and one part of #143 ask 3 (provenance decaying to `upstream` when a proposal copies a component from the library).

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
| [143](https://github.com/google-research/harness2/issues/143) | Enforcement gaps | ✅ fixed here (one part of 3 to do) | **1a**: the ledger marks each rollout live or dead (`live_R`, `dead` with errors); the report shows `dead rollouts N/R` and a warning. **1b**: one `is_absent_evidence()` covers every "no evidence" placeholder. **1c**: judge and report re-check R = 4k / P = 3k from the saved ledger and leave out picks that fail it. **2**: a reflection that keeps no lesson writes `NO_ENTRY.md` with its reason (`declined`, `malformed`, `session_failed`, `unverified`) and its edit rows; INDEX.md lists these under "Tried, no lesson kept". `confidence` is required and verdicts must be one of the four. **3**: lessons carry `provenance` (`measured` / `upstream` / `unverified`), read from their citations so the prompts stay identical to upstream; `unverified` lessons are not stored. Not yet: decaying provenance when a component is copied from the library. Tests: `dead_rollouts_are_listed_apart`, `a_reflection_without_a_lesson_leaves_a_record`, `provenance_reads_the_citations`, `verdicts_are_checked`. |
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
| [146](https://github.com/google-research/harness2/issues/146) / [144](https://github.com/google-research/harness2/issues/144) | `third_party/` missing from wheels / sdist | ✅ fixed here (our version) | Release tarballs and the Docker image ship `third_party/` and the dependency pins, and the binary looks next to itself when it isn't run from the source folder. |

## Features and proposals

| # | Upstream issue | Status | What harnessmith does |
|---|---|---|---|
| [4](https://github.com/google-research/harness2/issues/4) | `--max-usd` spend limit | ✅ done | A per-run cap (agents stop, run ends `capped`) and daily per-connection caps that fail over along the chain. |
| [5](https://github.com/google-research/harness2/issues/5) | Run on your own folder of tasks | ✅ done | Goal projects: type a goal or attach files in the UI, or `harness2 goal`, `harness2 solve`, the webhook, GitHub issues or schedules. |
| [161](https://github.com/google-research/harness2/issues/161) | Deterministic verification gate inside a k=1 trajectory | ✅ done (general form) | `llm/review.rs` checks deliverables against the goal before `finish` (file types, complete HTML, styling, depth, broken JSON/CSV, error text), with no model call. |
| [148](https://github.com/google-research/harness2/issues/148) / [147](https://github.com/google-research/harness2/issues/147) | Docker container runs as root | ✅ done | The Dockerfile runs as an unprivileged `harness` user. |
| [100](https://github.com/google-research/harness2/issues/100) / [64](https://github.com/google-research/harness2/issues/64) / [117](https://github.com/google-research/harness2/issues/117) | Antigravity, Pi, Claude Code, Hermes substrates | 🟢 partly | The native runtime works with any OpenAI-compatible, Anthropic, Gemini or Ollama model, and the improver can run through Claude Code (`claude_cli`). Dedicated Pi or Hermes substrates are not planned. |
| [94](https://github.com/google-research/harness2/issues/94) | Pin harness components the improver can't remove | ✅ done | **📌 Pin** on Base harness (project setting `pinned_paths`). A proposal that changes or removes a pinned file is rejected and retried. Test: `pinned_files_and_hidden_characters_are_refused`. |
| [96](https://github.com/google-research/harness2/issues/96) | Static check on an evolved harness | ✅ done | Goal-mode validation rejects zero-width, bidirectional-override and Unicode tag characters in any harness file, naming the code points. |
| [2](https://github.com/google-research/harness2/issues/2) | Opt-in human feedback for selection and lessons | ✅ done (first step) | **Prefer base1 / Prefer adopted** on the compare tab is stored with the goal and shown in Results. Preferring the base attempt saves a `human-feedback` memory note that later goals recall. Not yet: re-running the reflection from your choice. |
| [149](https://github.com/google-research/harness2/issues/149) | Use in production without Propose–Probe–Compose on every task | ✅ done | "Use as project base harness", memory promotion, A/B tests and shootouts, schedules, and now **base-harness versions with rollback**: each change keeps the previous version (40 kept), restorable from Base harness → Earlier versions. |
| [106](https://github.com/google-research/harness2/issues/106) | Publish a version / tag | 🔜 to do (ours) | Tag `v0.1.0`; the release workflow then builds binaries and a Docker image. |
| [57](https://github.com/google-research/harness2/issues/57) / [61](https://github.com/google-research/harness2/issues/61) / [55](https://github.com/google-research/harness2/issues/55) / [59](https://github.com/google-research/harness2/issues/59) | Outcome-based getting started; when to use it; credential-free example | ✅ done | [`docs/GETTING_STARTED.md`](GETTING_STARTED.md): what you get, when it helps (and when one run is enough), a first goal in 5 minutes with the offline model, the inspectable `harness2 demo` ledger, and where everything lives. |
| [179](https://github.com/google-research/harness2/issues/179) | Benchmark × executor catalog | ⚪ n/a | Upstream refactor (closed). |
| [136](https://github.com/google-research/harness2/issues/136) | Native subscription workflow for LAB | ⚪ n/a | Specific to upstream's LAB setup. |
| [178](https://github.com/google-research/harness2/issues/178), [91](https://github.com/google-research/harness2/issues/91), [71](https://github.com/google-research/harness2/issues/71), [39](https://github.com/google-research/harness2/issues/39) | Skill conversion, LAB study, analysis write-up, citation request | ⚪ n/a | Discussion, not code. |
| [73](https://github.com/google-research/harness2/issues/73), [152](https://github.com/google-research/harness2/issues/152), [155](https://github.com/google-research/harness2/issues/155) | Off-topic / spam | ⚪ n/a | |
| [194](https://github.com/google-research/harness2/issues/194) | Our announcement | — | |

## Upstream pull requests and the #193 audit

Fixes proposed in upstream PRs (or found by the [#193](https://github.com/google-research/harness2/issues/193) audit) that apply to this port.

| Upstream | Problem | Status | What harnessmith does |
|---|---|---|---|
| [#32](https://github.com/google-research/harness2/pull/32), audit 15, [#43](https://github.com/google-research/harness2/pull/43) | Symlinks let file tools or tree reads escape the workspace | ✅ fixed here | `safe_path` resolves the deepest existing part of a path and refuses it if a link leads outside the workspace (or is broken). Tree reads and fingerprints already skip links. Test: `symlinks_cannot_escape_the_workspace`. |
| [#37](https://github.com/google-research/harness2/pull/37), [#40](https://github.com/google-research/harness2/pull/40) | Agent commands inherit API keys | ✅ fixed here | Credential-looking variables are removed from every agent shell command and hook (`HARNESS2_PASS_ENV` to keep one). Test: `secrets_are_not_passed_to_commands`. |
| audit 16 | Proposer audits can contain secrets | ✅ fixed here | Mission, system prompt, response and error are redacted before they are written. Test: `keys_are_redacted`. |
| audit 10 | `claude_cli` runs with `--dangerously-skip-permissions` | ✅ fixed here | Fenced by default: `acceptEdits` in its temporary folder plus a short allowlist of commands. Opt out with `HARNESS2_CLAUDE_CLI_UNFENCED=1`. Allowing `python3` still lets a determined session run code, so the improver is best treated as trusted. |
| [#54](https://github.com/google-research/harness2/pull/54), audit 7 | Library writes race with parallel workers or concurrent runs | ✅ fixed here | Adding, forgetting and re-indexing take a per-domain file lock. |
| audit 2 | A retry's transport error hides why the earlier attempt was rejected | ✅ fixed here | The fallback reason keeps both. |
| [#164](https://github.com/google-research/harness2/pull/164), [#142](https://github.com/google-research/harness2/pull/142) | Non-string hook matchers or commands accepted | ✅ fixed here | Validation rejects a matcher that isn't a string or doesn't compile, and a command that isn't a string. At run time a bad matcher now skips only its own entry (it used to turn off every hook for the event). Test: `hook_matchers_and_commands_must_be_strings`. |
| [#169](https://github.com/google-research/harness2/pull/169) | Hyphenated top-level script names rejected | ✅ fixed here | `scripts/check-totals.py` is allowed, as the error message already promised. |
| [#69](https://github.com/google-research/harness2/pull/69), [#13](https://github.com/google-research/harness2/pull/13) | Incomplete or failed audit entries replayed from cache | ✅ fixed here | A cache hit needs a readable `meta.json` with `ok: true` and no error. |
| audit 4 | Base roll reported ok when the run folder is missing | 🟢 not affected | Goal-mode bases are saved by the server, not rolled from a run folder. |
| [#175](https://github.com/google-research/harness2/pull/175), [#70](https://github.com/google-research/harness2/pull/70), [#118](https://github.com/google-research/harness2/pull/118) | Answer keys / rubric leaking into model context | 🟢 not affected | The goal rubric stays in `task.json` and is only read by the judge; the library and context builders refuse grader verdicts (`assert_tree_has_no_grader_verdicts`). |
