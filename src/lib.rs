// Copyright 2026 Google LLC (original Python implementation)
// Rust port licensed under the Apache License, Version 2.0.

//! Harness²: Recursive Agent Harnessing for an Open World.
//!
//! A Rust port of <https://github.com/google-research/harness2>. Three axes stay separate:
//! method (`core`), substrate (the agent framework a solver rollout runs under) and benchmark
//! (a `BenchSpec` value plus an adapter). The `llm` module adds a model-agnostic agent runtime
//! so any OpenAI-compatible, Anthropic, Gemini or Ollama endpoint can act as solver, improver
//! or selector; `ui` serves a web interface for connecting models and giving the system a goal.

pub mod benchmarks;
pub mod cli;
pub mod config;
pub mod core;
pub mod llm;
pub mod log;
pub mod substrate;
pub mod ui;
pub mod util;
