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
//! ### C++ Interoperability
//!
//! The build script will read the `CXX` environment variable to choose which C++ compiler to use
//! for Rust/C++ interop. Note that the crate itself does not currently use C++ code, only examples
//! do.
//!
//! #### Known issues
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
        include!("poc-kokkos-rs/src/include/hello.hpp");

        fn say_hello();

        fn say_many_hello();
    }
}

pub mod functor;
pub mod routines;
pub mod view;
