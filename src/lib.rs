//! # Kokkos-rs: A Proof-of-Concept
//!
//! ## Scope of the Project
//!
//! The goal of this project is not to produce an entire Kokkos implementation nor to
//! replicate the existing C++ library. While the current C++ source code is interesting
//! to use as inspiration, the main reference is the model description.
//!
//! Additionally, because of language specific features (Rust strict compilation rules,
//! C++ templates), you can expect the underlying implementation of concepts to be
//! vastly different.
//!
//!
//! ## Quickstart
//!
//! The PoC itself is a library, but you can run benchmarks and examples out of the box:
//!
//! ```bash
//! # all benchmarks
//! cargo bench
//! # a specific benchmark
//! cargo bench --bench <BENCHMARK>
//! # a specific example
//! cargo run --example <EXAMPLE>
//! ```
//!
//! Generate local documentation:
//!
//! ```bash
//! cargo doc --no-deps --open
//! ```
//!
//! Note that some elements of the documentation are feature specific.
//!
//! ## Compilation
//!
//! ### Features
//!
//! Using `features`, the crate can be compiled to use different backend for execution of parallel section.
//! These can also be enabled in benchmarks.
//!
//! ```bash
//! cargo build --features <FEATURE>
//! ```
//!
//! Available features:
//!
//! - `rayon`: Uses the [rayon][2] crate to handle parallelization on CPU.
//! - `threads` : Uses [`std::thread`] methods to handle parallelization on CPU.
//! - `gpu`: Currently used as a way to gate GPU usage as this cannot be done in pure Rust.
//!
//! [1]: https://kokkos.github.io/kokkos-core-wiki/index.html
//! [2]: https://docs.rs/rayon/latest/rayon/

#![allow(incomplete_features)]
#![feature(min_generic_const_args)]
#![feature(adt_const_params)]

pub mod functor;
pub mod view;
