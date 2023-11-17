//! view parameterization code
//!
//! This module contains all code used to parameterize views. Some parameters are direct
//! Kokkos replicates while others are Rust-specific. Currently supported parameters
//! include:
//!
//! - Type of the data owned by the view (Rust-specific)
//! - Memory layout
//!
//! Possible future implementations include:
//!
//! - Memory space
//! - Memory traits?
//!
//!

use std::fmt::Debug;

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
use atomic::Atomic;

/// Maximum possible depth (i.e. number of dimensions) for a view.
pub const MAX_VIEW_DEPTH: usize = 8;

pub trait DataTraits: Debug + Clone + Copy + Default {}

impl DataTraits for f64 {}
impl DataTraits for f32 {}

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
pub type InnerDataType<T> = T;

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
pub type InnerDataType<T> = Atomic<T>;

#[derive(Debug)]
/// Enum used to identify the type of data the view is holding. See variants for more
/// information. The policy used to implement the [PartialEq] trait is based on Kokkos'
/// [`equal` algorithm](https://kokkos.github.io/kokkos-core-wiki/API/algorithms/std-algorithms/all/StdEqual.html).
pub enum DataType<'a, T>
where
    T: DataTraits,
{
    /// The view owns the data.
    Owned(Vec<InnerDataType<T>>),
    /// The view borrows the data and can only read it.
    Borrowed(&'a [InnerDataType<T>]),
    /// The view borrows the data and can both read and modify it.
    MutBorrowed(&'a mut [InnerDataType<T>]),
}

/// Kokkos implements equality check by comparing the pointers, i.e.
/// two views are "equal" if and only if their data field points to the
/// same memory space.
impl<'a, T> PartialEq for DataType<'a, T>
where
    T: DataTraits,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // are the deref operations necessary?
            // this one is technically necessary because self==self should return true
            (Self::Owned(l0), Self::Owned(r0)) => (*l0).as_ptr() == (*r0).as_ptr(),
            // compare pointers
            // deref Owned only once, twice the others
            (Self::Owned(l0), Self::Borrowed(r0)) => (*l0).as_ptr() == (**r0).as_ptr(),
            (Self::Owned(l0), Self::MutBorrowed(r0)) => (*l0).as_ptr() == (**r0).as_ptr(),
            (Self::Borrowed(l0), Self::Owned(r0)) => (**l0).as_ptr() == (*r0).as_ptr(),
            (Self::MutBorrowed(l0), Self::Owned(r0)) => (**l0).as_ptr() == (*r0).as_ptr(),

            (Self::Borrowed(l0), Self::Borrowed(r0)) => (**l0).as_ptr() == (**r0).as_ptr(),
            (Self::Borrowed(l0), Self::MutBorrowed(r0)) => (**l0).as_ptr() == (**r0).as_ptr(),
            (Self::MutBorrowed(l0), Self::MutBorrowed(r0)) => (**l0).as_ptr() == (**r0).as_ptr(),
            (Self::MutBorrowed(l0), Self::Borrowed(r0)) => (**l0).as_ptr() == (**r0).as_ptr(),
        }
    }
}

/// Enum used to represent data layout. Struct enums is used in order to increase
/// readability.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Layout<const N: usize> {
    /// Highest stride for the first index, decreasing stride as index increases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    #[default]
    Right,
    /// Lowest stride for the first index, increasing stride as index decreases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    Left,
    /// Custom stride for each index. Must be compatible with dimensions.
    Stride { s: [usize; N] },
}

/// Compute correct strides of each index using dimensions and specified layout.
pub fn compute_stride<const N: usize>(dim: &[usize; N], layout: &Layout<N>) -> [usize; N] {
    assert_eq!(N.clamp(1, MAX_VIEW_DEPTH), N); // 1 <= N <= MAX_N
    match layout {
        Layout::Right => {
            let mut stride = [1; N];

            let mut tmp: usize = 1;
            for i in (1..N).rev() {
                tmp *= dim[i];
                stride[N - i] = tmp;
            }

            stride.reverse();
            stride
        }
        Layout::Left => {
            let mut stride = [1; N];

            let mut tmp: usize = 1;
            for i in 0..N - 1 {
                tmp *= dim[i];
                stride[i + 1] = tmp;
            }

            stride
        }
        Layout::Stride { s } => *s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_right() {
        // dim = [n0, n1, n2, n3]
        let dim = [3, 4, 5, 6];

        let cmp_stride = compute_stride(&dim, &Layout::Right);
        // n3 * n2 * n1, n3 * n2, n3, 1
        let ref_stride: [usize; 4] = [6 * 5 * 4, 6 * 5, 6, 1];

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn stride_left() {
        // dim = [n0, n1, n2, n3]
        let dim = [3, 4, 5, 6];

        let cmp_stride = compute_stride(&dim, &Layout::Left);
        // 1, n0, n0 * n1, n0 * n1 * n2
        let ref_stride: [usize; 4] = [1, 3, 3 * 4, 3 * 4 * 5];

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn one_d_stride() {
        // 1d view (vector) of length 1
        let dim: [usize; 1] = [8];
        let ref_stride: [usize; 1] = [1];
        let mut cmp_stride = compute_stride(&dim, &Layout::Right);
        assert_eq!(ref_stride, cmp_stride);
        cmp_stride = compute_stride(&dim, &Layout::Left);
        assert_eq!(ref_stride, cmp_stride);
    }
}
