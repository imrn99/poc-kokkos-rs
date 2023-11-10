# Kokkos-rs: A Proof-of-Concept

## Motivation

A number of Rust's built-in features seems compatible or even complementary to 
the programming model defined by [Kokkos][1]. This project serve as partial 
proof and verification of that statement.


## Scope of the Project

~~The main focus of this Proof-of-Concept is the architecture and approach used by
Kokkos for data management. While multiple targets support (Serial, [rayon][2], OpenMP)
could be interesting, it is not the priority.~~

Rudimentary data structure implementation being done, the goal is now to write a simple
program using a `parallel_for` statement with satisfying portability as defined by Kokkos.

Additionally, some features of Kokkos are not reproducible in Rust (GPU targetting, 
templating); These create limits for the implementation that may or may not be bypassed.
This makes limit-testing an fundamental part of the project.


## Quickstart

### Example

The PoC itself is a library, but you can run examples by using the following command: 

```
cargo run --example hello-world
```

The following examples are available: 

- `hello-world`: ...
- `openmp-parallel`: ...


### Documentation

A consise documentation can be generated and accessed using the following command: 

```
cargo doc --open --no-deps
```

## References

### View Implementation

- `ndarray` Rust implementation: [link][NDARRAY]
- Const generics documentation from The Rust Reference: [link][CONSTG]
- `move` keyword semantic & implementation: [link][MOVE]


### Functor Implementation

- A very specific answer to a very specific rust-lang issue: [link][FNIMPL]



[1]: https://kokkos.github.io/kokkos-core-wiki/index.html
[2]: https://docs.rs/rayon/latest/rayon/

[NDARRAY]: https://docs.rs/ndarray/latest/ndarray/
[CONSTG]: https://doc.rust-lang.org/reference/items/generics.html
[FNIMPL]: https://github.com/rust-lang/rust/issues/29625#issuecomment-1692602873
[MOVE]: https://stackoverflow.com/questions/30288782/what-are-move-semantics-in-rust