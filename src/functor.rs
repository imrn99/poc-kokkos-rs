//!
//!
//!
//!

//pub trait KernelArgs {}

//impl KernelArgs for usize {}

//impl<const N: usize> KernelArgs for [usize; N] {}

/// Until some work is done to have a better solution[^sol1][^sol2], this will
/// be an enum and kernels will be written in an idiomatic way.
///
/// [^sol1]: Current tracking issue for upcasting implementation: <https://github.com/rust-lang/rust/issues/65991>
///
/// [^sol2]: Current tracking issue to allow impl trait usage in types aliases: <https://github.com/rust-lang/rust/issues/63063>
pub enum KernelArgs<const N: usize> {
    Index1D(usize),
    IndexND([usize; N]),
    Handle,
}

#[cfg(feature = "rayon")]
pub type ForKernel<'a, const N: usize> = Box<dyn Fn(KernelArgs<N>) + Send + Sync + 'a>;

#[cfg(not(any(feature = "rayon", feature = "threads")))]
pub type ForKernel<'a, const N: usize> = Box<dyn FnMut(KernelArgs<N>) + 'a>;
