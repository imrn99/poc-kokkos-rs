//! # Kokkos-rs: A Proof-of-Concept
//!
//! ## Scope of the Project
//!
//! The main focus of this Proof-of-Concept is the architecture and approach used by
//! [Kokkos][1] for data management. While multiple targets support (Serial, [rayon][2],
//! OpenMP) could be interesting, it is not the priority.
//!
//! Additionally, some features of Kokkos are not reproducible in Rust (GPU targetting,
//! templating); These create limits for the implementation that may or may not be bypassed.
//!
//!
//! ## Quickstart
//!
//!
//!
//! ## Compilation
//!
//!
//! [1]: https://kokkos.github.io/kokkos-core-wiki/index.html
//! [2]: https://docs.rs/rayon/latest/rayon/

pub mod view;
