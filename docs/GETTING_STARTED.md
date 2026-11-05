# Getting started

## What you get

You type a goal ("write a research document on dogs", "turn this CSV into a one-page brief", "make a sleek landing page"). Harnessmith runs an agent on it several times. An improver model reads those attempts and edits the agent's harness (its instructions, guardrails, skills and helper scripts) to do this kind of goal better. A selector then picks the best output without knowing which harness produced it. You get:

- the deliverables, as files you can open, download or reveal in your file manager;
- the adopted harness, which you can keep as the project's starting point;
- a lesson in the project's library, which later goals of the same kind read.

## When it helps, and when it doesn't

**It helps** when a goal is worth more than one attempt: a deliverable with requirements to meet (format, sections, numbers that must add up), a kind of task you'll repeat, or a model that does the work but keeps missing details. The improver turns those misses into instructions and scripts, and the library carries them to the next goal.

**A single run is enough** for one-off questions or quick edits. Use `harness2 solve "<goal>"`, or a goal with budget k = 1 and no improver, when you only want an answer. A full run costs about 4 solver attempts and 4 improver sessions at k = 1.

## Five minutes, no keys: the offline model

Everything below runs on your machine, with no account or API key. The `mock` model is a stand-in that produces placeholder output instantly, so you can see the whole pipeline before connecting a real model.

```bash
cargo build --release
```

```bash
./target/release/harness2 models add offline --provider mock --default-model mock-1
```

```bash
./target/release/harness2 serve
```

Open http://127.0.0.1:8765:
1. **Models → Roles:** set solver, improver and selector to `offline/mock-1`, then **Save roles**.
2. **Run a goal:** type "Summarise the attached file into output/answer.md", optionally attach any small file, and press **Start**.
3. **Watch** the pipeline map: base run, then propose and probe (INV, DOM), compose, select and learn. Open **Overview** to see the live figure.
4. **History → the goal:** see the output, **compare** the base attempt with the adopted one, open **files**, and download.

For a slower run you can follow step by step, set `HARNESS2_MOCK_DELAY_MS=2500` before `serve`.

The same run without the UI, for an inspectable ledger on disk:

```bash
./target/release/harness2 demo --output runs/demo --k 1
```

It writes the full layout (rollouts, harness proposals, selection, ledger) under `runs/demo` with deterministic local editors, so you can read every file the method produces.

## Next: a real model

- **Local:** install [Ollama](https://ollama.com), then pull a model that fits your machine. **Models → Model advisor** tells you which one, and can download it.
- **API:** **Models → Add a connection** (OpenAI-compatible, Anthropic, Gemini, OpenRouter…), then **Test**.
- **Mixed:** a local solver with an API improver usually gives the best result per dollar. The advisor offers it as a one-click setup.

Then use **✨ Improve with AI** on your goal to turn a one-liner into a full brief with grading criteria, and **Start**.

## Where things are

| You want | Go to |
|---|---|
| the output files | History → goal → **files** (or **Open folder**) |
| whether adapting helped | **Results** (base vs adopted scores, when the goal has grading criteria) |
| what the agent did, step by step | History → goal → click a stage on the map |
| why something failed | **Doctor**, and the red / amber notes on the map |
| to keep a good harness | History → goal → **harness** → **Use as project base harness** (older versions stay under Base harness → Earlier versions) |
