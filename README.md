# Kokkos-rs: A Proof-of-Concept

## Motivation

A number of Rust's built-in features seems compatible or even complementary to 
the programming model defined by [Kokkos][1]. This project serve as partial 
proof and verification of that statement.


## Scope of the Project

The goal of this project is not to produce an entire Kokkos implementation nor to
replicate the existing C++ library. While the current C++ source code is interesting
to use as inspiration, the main reference is the model description. 

Additionally, because of language specific features (Rust strict compilation rules, 
C++ templates), you can expect the underlying implementation of concepts to be 
vastly different.

## Quickstart

The PoC itself is a library, but you can run benchmarks and examples out of the box.

### Benchmarks

Benchmarks can be run using the following command:

```bash
# all benchmarks
cargo bench
# a specific benchmark
cargo bench --bench <BENCHMARK>
```

All results are compiled to the `target/criterion/` folder. The following
benchmarks are available:

**Layout:**
- `layout-comparison`: Bench a Matrix-Matrix product three times, using the worst possible layout,
  the usual layout, and then the optimal layout for the operation. This shows the importance of layout
  selection for performances.
- `layout-size`: Bench a Matrix-Matrix product using the usual layout and the optimal layout,
  over a range of sizes for the square matrices. This shows the influence of cache size over
  layout importance.
**Computation:**
- `axpy` / `gemv` / `gemm`: Measure speedup on basic BLAS implementations by running the same kernel
  in serial mode first, then using parallelization on CPU. _Meant to be executed using features_.
- `hardcoded_gemm`: Compute the same operations as the `gemm` benchmark, but using a hardcoded implementation
  instead of methods from the PoC. Used to assess the additional cost induced by the library.
**Library overhead:**
- `view_init`: Compare initialization performances of regular vectors to Views; This
  is used to spot potential scaling issues induced by the more complex structure of Views.
- `view_access`: Compare data access performances of regular vectors to Viewsview; This
  is used to spot potential scaling issues induced by the more complex structure of Views.

Additionally, a kokkos-equivalent of the blas kernels can be found in the `blas-speedup-kokkos/`
subdirectory. These are far from being the most optimized implementation, instead they are written
as close-ish counterparts to the Rust benchmarks.


### Examples

```bash
cargo run --example <EXAMPLE>
```

The following examples are available:

- `hello_world`: ...
- `hello_world_omp`: ...


## Features

Using `features`, the crate can be compiled to use different backend for execution of parallel section.
These can (and should) also be enabled in benchmarks.

```bash
cargo build --features <FEATURE>
```

Available features:

- `rayon`: Uses the [rayon][2] crate to handle parallelization on CPU.
- `threads` : Uses `std::thread` methods to handle parallelization on CPU.
- `gpu`: Currently used as a way to gate GPU usage as this cannot be done in pure Rust.

## Compilation

The build script will read the `CXX` environment variable to choose which C++ compiler to use
for Rust/C++ interop. Note that the crate itself does not currently use C++ code, only examples
do.

## References

- The Kokkos Wiki: [link][1]
- `rayon` crate documentation: [link][2]


[1]: https://kokkos.github.io/kokkos-core-wiki/index.html
[2]: https://docs.rs/rayon/latest/rayon/
