//! functor & kernel related code
//!
//! This module contains all functor and kernel related code. Its content
//! is highly dependant on the features enabled since the traits that a
//! kernel must satisfy changes totally depending on the backend used.

#[cfg(doc)]
use crate::routines::parameters::RangePolicy;

/// Kernel argument enum
///
/// In the Kokkos library, there is a finite number of kernel signatures.
/// Each is associated to/determined by a given execution policy.
/// In order to have kernel genericity in Rust, without introducing overhead
/// due to downcasting, the solution was to define kernel arguments as a
/// struct-like enum.
///
/// ### Example
///
/// One-dimensional kernel:
/// ```
/// // Range is defined in the execution policy
/// let kern = |arg: KernelArgs<1>| match arg {
///         KernelArgs::Index1D(i) => {
///             // body of the kernel
///             println!("Hello from iteration {i}")
///         },
///         KernelArgs::IndexND(_) => unimplemented!(),
///         KernelArgs::Handle => unimplemented!(),
///     };
/// ```
///
/// 3D kernel:
/// ```
/// // Use the array
/// let kern = |arg: KernelArgs<3>| match arg {
///         KernelArgs::Index1D(_) => unimplemented!(),
///         KernelArgs::IndexND(idx) => { // idx: [usize; 3]
///             // body of the kernel
///             println!("Hello from iteration {idx:?}")
///         },
///         KernelArgs::Handle => unimplemented!(),
///     };
///
/// // Decompose the array
/// let kern = |arg: KernelArgs<3>| match arg {
///         KernelArgs::Index1D(_) => unimplemented!(),
///         KernelArgs::IndexND([i, j, k]) => { // i,j,k: usize
///             // body of the kernel
///             println!("Hello from iteration {i},{j},{k}");
///         },
///         KernelArgs::Handle => unimplemented!(),
///     };
/// ```
pub enum KernelArgs<const N: usize> {
    /// Arguments of a one-dimensionnal kernel (e.g. a [RangePolicy][RangePolicy::RangePolicy]).
    Index1D(usize),
    /// Arguments of a `N`-dimensionnal kernel (e.g. a [MDRangePolicy][RangePolicy::MDRangePolicy]).
    IndexND([usize; N]),
    /// Arguments of a team-based kernel.
    Handle,
}

cfg_if::cfg_if! {
    if #[cfg(feature = "rayon")] {
        /// `rayon`-specific kernel type.
        pub type ForKernelType<'a, const N: usize> = Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>;
    } else if #[cfg(feature = "threads")] {
        /// Standard threads specific kernel type.
        pub type ForKernelType<'a, const N: usize> = Box<dyn Fn(KernelArgs<N>) + Send + 'a>;
    } else {
        /// Fall back kernel type.
        pub type ForKernelType<'a, const N: usize> = SerialForKernelType<'a, N>;
    }
}

/// Serial kernel type.
pub type SerialForKernelType<'a, const N: usize> = Box<dyn FnMut(KernelArgs<N>) + 'a>;
