//! # Kokkos-rs: A Proof-of-Concept
//!
//! ## Scope of the Project
//!
//! ~~The main focus of this Proof-of-Concept is the architecture and approach used by
//! [Kokkos][1] for data management. While multiple targets support (Serial, [rayon][2],
//! OpenMP) could be interesting, it is not the priority.~~
//!
//! Rudimentary data structure implementation being done, the goal is now to write a simple
//! program using a `parallel_for` statement with satisfying portability as defined by Kokkos.
//!
//! Additionally, some features of Kokkos are not reproducible in Rust (GPU targetting,
//! templating); These create limits for the implementation, hence the existence of this PoC.
//! This makes limit-testing an fundamental part of the project.
//!
//!
//! ## Quickstart
//!
//! The PoC itself is a library, but you can run benchmarks and examples out of the box.
//!
//! ### Benchmarks
//!
//! Benchmarks can be run using the following command:
//!
//! ```bash
//! # all benchmarks
//! cargo bench
//! # a specific benchmark
//! cargo bench --bench bench_name
//! ```
//!
//! All results are compiled to the `target/criterion/` folder. The following
//! benchmarks are available:
//!
//! - `layout`: Matrix-Vector product computation; This is used to put numbers on the
//!   importance of data layout in memory.
//! - `view_init`: Compare initialization performances of regular vectors to [Views][view]; This
//!   is used to spot potential scaling issues induced by the more complex structure of Views.
//! - `view_access`: Compare data access performances of regular vectors to [Views][view]; This
//!   is used to spot potential scaling issues induced by the more complex structure of Views.
//! - `mdrange_populate`: Compare performance of our implementation of MDRangePolicy compared to
//!   regular implementation. Currently, only a serial implementation with no tiling is tested.
//! - `feature`: Assess the correct usage of feature-specific backend. This one is meant to be run
//!   multiple times, with varying features each time (e.g. no feature, then `rayon` to observe the
//!   speedup).
//!
//!
//! ### Examples
//!
//! ```bash
//! cargo run --example hello-world
//! ```
//!
//! The following examples are available:
//!
//! - `hello_world`: ...
//! - `hello_world_omp`: ...
//!
//!
//! ## Features
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
//! ## Compilation
//!
//! The build script will read the `CXX` environment variable to choose which C++ compiler to use
//! for Rust/C++ interop. Note that the crate itself does not currently use C++ code, only examples
//! do.
//!
//! ### Known issues
//!
//! - On MacOs: Does not work with Apple Clang
//!   - Solution: Homebrew Clang or tinker with flags to get OpenMP to work
//! - On MacOs: XCode 15.0 was shipped with a broken `ld`
//!   - Solution: pass the flag `-ld_classic` to the linker. Note that this flag isn't
//!     recognized by the `ld` of previous versions of XCode. The line needed is
//!     written in the `build.rs` script, it just needs to be uncommented if necessary.
//!
//! [1]: https://kokkos.github.io/kokkos-core-wiki/index.html
//! [2]: https://docs.rs/rayon/latest/rayon/

//#![feature(type_alias_impl_trait)]

#[cxx::bridge(namespace = "")]
/// C++ inter-op code
pub mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("poc-kokkos-rs/include/hello.hpp");

        fn say_hello();

        fn say_many_hello();
    }
}

pub mod functor;
pub mod routines;
pub mod view;
