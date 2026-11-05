//! A model-agnostic runtime: named model connections, one chat client for every supported
//! wire protocol, and a tool-using agent loop that both the native solver substrate and the
//! native `api` proposer channel run on.

pub mod agent;
pub mod client;
pub mod connections;
pub mod approval;
pub mod gate;
pub mod mcp;
pub mod health;
pub mod memory;
pub mod review;
pub mod sandbox;
pub mod session;
