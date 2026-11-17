//! Lucy core: a sparse, event-driven substrate of cells that only talk to neighbours.
//! Cost scales with cells that fire, never with total graph size (all decay is lazy).

pub mod buffer;
pub mod kernel;
pub mod num;
pub mod observe;
pub mod rules;
pub mod scheduler;
pub mod seed;
pub mod senses;
pub mod store;
