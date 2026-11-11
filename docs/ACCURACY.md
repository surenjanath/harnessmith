# Accurate answers from small models

A small model asked "what is the total revenue in this CSV?" will usually reply with a number that looks right and is not. Harnessmith's **accuracy boost** changes how such a question is answered, and measures the difference.

RESULTS_PLACEHOLDER

## What the harness does differently

### 1. The model writes a program instead of an answer

A goal that asks for one exact answer (a number, a name, a date, a count) is not sent into the open-ended agent loop. The harness drives a fixed sequence, one narrow request at a time:

1. **Write a program.** The model is shown the question and its files and asked for one short Python program that prints the answer. The instructions are written for the files actually there: the exact line that loads each one, its columns or keys, what one line of a log looks like, and how to combine two tables that share a column.
2. **Run it.** The program runs in the sandbox, behind the operator's gate, with the workspace as its working directory. It runs like a notebook cell, so the habits of small models do not cost an attempt: the standard modules are already imported, the value of a last bare expression is printed, and `input()` fails at once instead of hanging.
3. **Repair it.** A failure comes back as the failing line and the exception, plus the one hint that usually fixes that kind of error. The model gets one chance to send the corrected program.
4. **Read the answer off the output.** The answer is what the program printed, never the model's recollection of it. A program that prints "not found" or `None` has not answered. If the question asks for cents and the program printed a whole number, it is sent back once.

Arithmetic, counting, sorting, date differences and file reading are done by Python. The model only has to say *what* to compute.

### 2. Independent attempts vote

The sequence is repeated from scratch (3 times by default). The first attempt is the model's best guess; later ones are sampled for variety and are told what made earlier attempts fail. The harness compares the answers by value, so `6263.75`, `$6,263.75` and `The total is 6263.75` are the same answer, and takes the one most attempts agree on. Two attempts that agree with nothing against them end the voting early.

The agreement is the **confidence** you are shown with the answer:

| confidence | meaning |
|---|---|
| high | two or more attempts agree and they are the majority |
| medium | one program ran and nothing contradicted it, or a plurality without a majority |
| low | the attempts disagree; the answer is a best guess and says so |

### 3. When attempts disagree

One step-by-step answer without a program is taken as a tie-break. If that settles nothing and the project names a stronger model under **If attempts disagree, ask**, that model is asked once and its answer stands. This makes "small model first, large model only when unsure" a setting.

### 4. Experience: verified answers become worked examples

When two or more computed attempts agree and none disagrees, the question, the shape of its files and the program are kept in the project's experience (`experience.json`). The next question of the same kind gets that program as a worked example. A small model adapts a solution far more reliably than it invents one.

### 5. Sleep: consolidate and dream

This part follows ["Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"](https://arxiv.org/abs/2606.03979) (Behrouz, Hashemi, Javanmard and Mirrokni, Google Research, 2026). The paper splits a learner's life into a wake phase and an offline sleep phase with two stages: *memory consolidation*, and *dreaming*, where the model generates its own practice material and keeps what has training value. The paper trains weights. A harness cannot, so the same cycle is run on what the harness owns:

| In the paper | In Harnessmith |
|---|---|
| Wake: the model works and gathers new knowledge | Goals are answered; verified solutions are added to experience |
| Consolidation: knowledge moves into long-term capacity | One example is kept per kind of question, examples that never helped are forgotten, unused memory notes fade |
| Dreaming: the model generates practice data and rehearses | Each example becomes a **new practice task**: its input data is changed (rows shuffled and dropped, numbers altered) or the numbers in the question are changed together with the program |
| Synthetic examples are filtered by their training value | The right answer to a practice task is computed by **running the verified program on the changed data**. The model is asked cold; if it fails, again with the example. An example the model needs is kept and ranked up; one that never helps is forgotten |
| Self-generated curriculum | **Explore**: the model asks itself new questions about your files; answers that every attempt agrees on are kept, so a kind of question is solved before you ask it |

Ground truth in rehearsal never comes from the model's opinion of its own answer. It comes from a program that was already verified, which is what stops a dream from reinforcing a mistake.

Run it with **🌙 Sleep now** on the Memory page, with `harness2 sleep --project <name>`, or turn on **Sleep on its own when idle**. Each sleep writes a journal: what was consolidated, every practice task, and whether the example was needed.

### 6. In every other goal: compute, and show where figures come from

Goals that are not one exact question (a report, a page, a document) still run on the agent loop. With the boost on, the agent also gets:

- a `python` tool (same notebook-style runner) and a `calc` tool for plain arithmetic;
- small input files shown with the task, so its first step is work and not a read;
- a fenced Python block it writes *instead of* a tool call is run, not ignored;
- a **grounding check** before it may finish: every figure in the written deliverable must appear in the goal, the input files or the output of a tool call in the session. Figures that match nothing are handed back with "compute them". This catches the plainest failure there is, a number that was never calculated.

### 7. Pages and coding tasks are built, checked and compared

Two more kinds of goal leave the open-ended agent loop when the boost is on, because a small model loses a whole file to one JSON-escaping slip, or never tests what it wrote:

- **A single web page** ("a sleek hello world page", "a landing page for…"). The model is asked for the complete file in one fenced block, with a design brief. The harness writes the file and reads it against a checklist of 14 points: doctype, complete document, title, viewport, real CSS, flex or grid layout, typography, CSS variables, background, rounded corners and shadow, a transition or animation, responsive sizing, no placeholder text, and the content the goal asks for. The misses go back to the model once as a list. Up to three candidates are made, and the one that passes the most checks is kept.
- **A Python coding task** ("write a function…", "coding challenge: …"). The model writes a solution in one fenced block and then assert-style tests for it, a few times over. **Every candidate is run against every test**, its own and the other candidates', so one wrong test cannot crown a wrong solution. A test no candidate passes is dropped as probably wrong. The best candidate gets one repair on the tests it fails, and the candidate that passes the most tests is written to `output/solution.py`, with the tests it passes listed in `output/answer.md`.

Both report what passed and what is still open, and a result that passes everything counts as verified.

### 8. A verified answer skips the improvement loop

In a full run, when the first attempt's answer is verified with high confidence, it is adopted as it stands. The harness-improvement loop exists to fix what went wrong, and nothing did. An exact question in the UI therefore takes about as long as the attempts themselves.

## Which goals take the exact-answer path

A goal goes to the answer engine when it asks a question ("how many", "what is", "which … has", "calculate", "find the", "the total / average / median", and so on) and does not ask for something to be written or built ("write a report", "create a page", "explain", "summarise", "compare"), and names no output file type. A goal with no input files that asks for one web page, or for a Python function, program or solution to a coding challenge, goes to the maker (section 7). Everything else runs on the agent loop. If an engine produces nothing, the goal falls back to the agent loop as well.

## Settings

Models → Roles for this project:

| setting | default | what it does |
|---|---|---|
| Accuracy boost | Auto | **Auto** turns it on for local servers (Ollama, llama.cpp, LM Studio…) and for models whose name gives a size up to 40B or says mini / small / tiny / nano / lite. **On** applies it to every model, **Off** to none. |
| Attempts to vote over | 3 | 1 turns voting off (one program, medium confidence at best). 5 suits very small models. |
| If attempts disagree, ask | – | A stronger model, asked once, only when the vote is split. |
| Learn from verified answers | On | Keep verified solutions as worked examples. |

Memory → Experience: **Sleep on its own when idle**, practice tasks per sleep, new questions to explore.

The same settings are in `project.json` as `accuracy`, `accuracy_samples`, `escalate_model`, `experience`, `settle_verified`, `sleep_auto`, `sleep_dreams`.

## Measure your own model

```bash
./target/release/harness2 accuracy --model ollama/qwen2.5:3b
```

or **Models → Accuracy check** in the UI. Options:

| option | meaning |
|---|---|
| `--modes plain,boost` | `plain`: the model alone, thinking step by step. `agent`: the agent loop with the boost off. `boost`: the boost. `learned`: the boost in a project that first worked through other questions of the same kinds (another seed) and slept. |
| `--per-category 2` | questions per kind; there are 14 kinds |
| `--seed 1` | another seed gives other questions with other answers |
| `--samples 3` | attempts the boost votes over |
| `--escalate <model>` | the stronger model for split votes |
| `--only csv_total,dates` | only these kinds |

The questions are generated, so the right answers are computed and no model has seen them: warehouse arithmetic, discount and tax, compound interest, date differences, letter counts, median and mean, ordering, unit conversion, totals and group questions over a CSV, counting and lookups in a log, sums over JSON, and a two-file join. Every mode is checked by the same rule, on the answer it states. Results are saved under `runs/accuracy/` and can be resumed.

## Limits

- The boost fixes *computation and lookup* errors. It does not give a model reasoning it does not have: when a small model sets a problem up wrongly the same way every time, the attempts agree on a wrong answer. A stronger model for split votes helps only when the vote is split.
- An agreed answer can still be wrong, and a wrong one that is kept as experience will be reused. Examples are listed on the Memory page, where you can archive or delete them; sleep forgets the ones that fail in rehearsal.
- Programs run on your machine. They run inside the sandbox when one is available (bubblewrap or `unshare`), without network unless the project allows it, and behind the operator's gate. With the sandbox off they run as shell commands do.
- The exact-answer path needs `python3` and the shell tool allowed. Without them, goals run on the agent loop as before.
- Voting costs attempts: usually two or three short calls per question, which on a local model is seconds to a couple of minutes.
