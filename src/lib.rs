//! # Kokkos-rs: A Proof-of-Concept
//!
//! ## Scope of the Project
//!
//! The main focus of this Proof-of-Concept is the architecture and approach used by
//! [Kokkos][1] for data management. While multiple targets support (Serial, [rayon][2],
//! OpenMP) could be interesting, it is not the priority.
//!
//! Additionally, some features of Kokkos are not reproducible in Rust (GPU targetting,
//! templating); These create limits for the implementation, hence the existence of this PoC.
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
//! - `gemv`: Matrix-Vector product computation; This is used to put numbers on the
//!   importance of data layout in memory.
//! - `view_init`: Compare initialization performances of regular vectors to [Views][view]; This
//!   is used to spot potential scaling issues induced by the more complex structure of Views.
//! - `view_access`: Compare data access performances of regular vectors to [Views][view]; This
//!   is used to spot potential scaling issues induced by the more complex structure of Views.
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
//! - `hello-world`: ...
//! - `openmp-parallel`: ...
//!
//!
//! ## Compilation
//!
//! ### Known issues
//!
//! - On MacOs: Does not work with Apple Clang
//!   - Solution: Homebrew Clang or tinker with flags to get OpenMP to work
//! - On MacOs: XCode 15.0 was shipped with a broken `ld`
//!   - Solution: pass the flag `-ld_classic` to the linker.  This flag isn't
//!     recognized by the `ld` of previous versions of XCode. Remove it from
//!     `build.rs` if necessary.
//!
//! [1]: https://kokkos.github.io/kokkos-core-wiki/index.html
//! [2]: https://docs.rs/rayon/latest/rayon/

#[cxx::bridge(namespace = "")]
pub mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("poc-kokkos-rs/include/hello.hpp");

        fn say_hello();

        fn say_many_hello();
    }
}

pub mod view;
