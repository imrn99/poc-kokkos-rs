//!
//!
//!
//!

///
pub trait KernelArgs {}

impl KernelArgs for usize {}

impl<const N: usize> KernelArgs for [usize; N] {}

#[cfg(feature = "rayon")]
///
pub trait ForKernel<Args>: Fn(Args) + Send + Sync
where
    Args: KernelArgs,
{
}

#[cfg(not(any(feature = "rayon", feature = "thread")))]
///
pub trait ForKernel<Args>: FnMut(Args)
where
    Args: KernelArgs,
{
}
