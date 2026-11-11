//! A model-agnostic runtime: named model connections, one chat client for every supported
//! wire protocol, and a tool-using agent loop that both the native solver substrate and the
//! native `api` proposer channel run on.

pub mod agent;
pub mod calc;
pub mod client;
pub mod connections;
pub mod experience;
pub mod export;
pub mod approval;
pub mod gate;
pub mod ground;
pub mod maker;
pub mod mcp;
pub mod precise;
pub mod pycode;
pub mod health;
pub mod memory;
pub mod review;
pub mod sandbox;
pub mod session;
pub mod sleep;
pub mod web;
