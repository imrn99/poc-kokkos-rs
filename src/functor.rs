//! functor & kernel related code
//!
//! This module contains all functor and kernel related code. Its content
//! is highly dependant on the features enabled since the traits that a
//! kernel must satisfy changes totally depending on the backend used.

/// Kernel argument types
///
/// Until some work is done to have a better solution[^sol1][^sol2], this will
/// be an enum and kernels will be written in an idiomatic way.
///
/// [^sol1]: Current tracking issue for upcasting implementation: <https://github.com/rust-lang/rust/issues/65991>
///
/// [^sol2]: Current tracking issue to allow impl trait usage in types aliases: <https://github.com/rust-lang/rust/issues/63063>
pub enum KernelArgs<const N: usize> {
    /// Arguments of a one-dimensionnal kernel (e.g. a RangePolicy).
    Index1D(usize),
    /// Arguments of a `N`-dimensionnal kernel (e.g. a MDRangePolicy).
    IndexND([usize; N]),
    /// Arguments of a team-based kernel.
    Handle,
}

#[cfg(feature = "rayon")]
/// Kernel type used by the parallel statements.
///
/// Backend-specific type for `rayon`.
pub type ForKernel<'a, const N: usize> = Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>;

#[cfg(feature = "threads")]
/// Kernel type used by the parallel statements.
///
/// Backend-specific type for `std::thread`.
pub type ForKernel<const N: usize> = Box<dyn FnOnce(KernelArgs<N>) + Send + 'static>;

#[cfg(not(any(feature = "rayon", feature = "threads")))]
/// Kernel type used by the parallel statements.
///
/// Backend-specific type for serial execution.
pub type ForKernel<'a, const N: usize> = Box<dyn FnMut(KernelArgs<N>) + 'a>;
