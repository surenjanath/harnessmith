//! A model-agnostic runtime: named model connections, one chat client for every supported
//! wire protocol, and a tool-using agent loop that both the native solver substrate and the
//! native `api` proposer channel run on.

pub mod agent;
pub mod client;
pub mod connections;
pub mod gate;
pub mod health;
pub mod memory;
pub mod session;
