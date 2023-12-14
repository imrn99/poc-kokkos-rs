//! functor & kernel related code
//!
//! This module contains all functor and kernel related code. Its content
//! is highly dependant on the features enabled since the traits that a
//! kernel must satisfy changes totally depending on the backend used.
//!
//! Kernel signatures are handled using `cargo` features. Using conditionnal
//! compilation, the exact trait kernels must implement are adjusted according
//! to the backend used to dispatch statements.
//!
//! In order to have actual closures match the required trait implementation,
//! the same mechanism is used to define operations on [`Views`][crate::view].

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
/// use poc_kokkos_rs::functor::KernelArgs;
///
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
/// use poc_kokkos_rs::functor::KernelArgs;
///
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
        /// `parallel_for` kernel type. Depends on enabled feature(s).
        ///
        /// This type alias is configured according to enabled feature in order to adjust
        /// the signatures of kernels to match the requirements of the underlying dispatch routines.
        ///
        /// ### Possible Values
        /// - `rayon` feature enabled: `Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>`
        /// - `threads` feature enabled: `Box<dyn Fn(KernelArgs<N>) + Send + 'a>`
        /// - no feature enabled: fall back to [`SerialForKernelType`][SerialForKernelType]
        ///
        /// Current version: `rayon`
        pub type ForKernelType<'a, const N: usize> = Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>;
    } else if #[cfg(feature = "threads")] {
        /// `parallel_for` kernel type. Depends on enabled feature(s).
        ///
        /// This type alias is configured according to enabled feature in order to adjust
        /// the signatures of kernels to match the requirements of the underlying dispatch routines.
        ///
        /// ### Possible Values
        /// - `rayon` feature enabled: `Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>`
        /// - `threads` feature enabled: `Box<dyn Fn(KernelArgs<N>) + Send + 'a>`
        /// - no feature enabled: fall back to [`SerialForKernelType`][SerialForKernelType]
        ///
        /// Current version: `threads`
        pub type ForKernelType<'a, const N: usize> = Box<dyn Fn(KernelArgs<N>) + Send + 'a>;
    } else {
        /// `parallel_for` kernel type. Depends on enabled feature(s).
        ///
        /// This type alias is configured according to enabled feature in order to adjust
        /// the signatures of kernels to match the requirements of the underlying dispatch routines.
        ///
        /// ### Possible Values
        /// - `rayon` feature enabled: `Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>`
        /// - `threads` feature enabled: `Box<dyn Fn(KernelArgs<N>) + Send + 'a>`
        /// - no feature enabled: fall back to [`SerialForKernelType`][SerialForKernelType]
        ///
        /// Current version: no feature
        pub type ForKernelType<'a, const N: usize> = SerialForKernelType<'a, N>;
    }
}

/// Serial kernel type. Does not depend on enabled feature(s).
///
/// This is the minimal required trait implementation for closures passed to a
/// `for_each` statement.
pub type SerialForKernelType<'a, const N: usize> = Box<dyn FnMut(KernelArgs<N>) + 'a>;
