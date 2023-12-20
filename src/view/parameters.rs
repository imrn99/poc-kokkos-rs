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
//! - Memory traits

use std::{
    alloc::{alloc, Layout},
    fmt::Debug,
};

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
use atomic::Atomic;

use super::ViewError;

/// Maximum possible depth (i.e. number of dimensions) for a view.
pub const MAX_VIEW_DEPTH: usize = 8;

/// Supertrait with common trait that elements of a View should implement.
pub trait DataTraits: Debug + Clone + Copy + Default {}

impl DataTraits for f64 {}
impl DataTraits for f32 {}

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
/// Generic alias for elements of type `T` of a View.
///
/// This alias automatically changes according to features to ensure thread-safety
/// of Views. There are two possible values:
///
/// - any feature enabled: `InnerDataType<T> = Atomic<T>`. By adding the atomic wrapping,
/// operations on views can be implemented using thread-safe methods.
/// - no feature enabled: `InnerDataType<T> = T`.
///
/// **Current version**: no feature
pub type InnerDataType<T> = T;

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
/// Generic alias for elements of type `T` of a View.
///
/// This alias automatically changes according to features to ensure thread-safety
/// of Views. There are two possible values:
///
/// - any feature enabled: `InnerDataType<T> = Atomic<T>`. By adding the atomic wrapping,
/// operations on views can be implemented using thread-safe methods.
/// - no feature enabled: `InnerDataType<T> = T`.
///
/// **Current version**: thread-safe
pub type InnerDataType<T> = Atomic<T>;

// ~~~~~~~~~ Layouts ~~~~~~~~~

/// Enum used to represent data layout. Struct enums is used in order to increase
/// readability.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MemoryLayout<const N: usize> {
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
pub fn compute_stride<const N: usize>(dim: &[usize; N], layout: &MemoryLayout<N>) -> [usize; N] {
    assert_eq!(N.clamp(1, MAX_VIEW_DEPTH), N); // 1 <= N <= MAX_N
    match layout {
        MemoryLayout::Right => {
            let mut stride = [1; N];

            let mut tmp: usize = 1;
            for i in (1..N).rev() {
                tmp *= dim[i];
                stride[N - i] = tmp;
            }

            stride.reverse();
            stride
        }
        MemoryLayout::Left => {
            let mut stride = [1; N];

            let mut tmp: usize = 1;
            for i in 0..N - 1 {
                tmp *= dim[i];
                stride[i + 1] = tmp;
            }

            stride
        }
        MemoryLayout::Stride { s } => *s,
    }
}

// ~~~~~~~~~ Memory space ~~~~~~~~~

pub enum MemorySpace {
    CPU,
    GPU,
}

pub fn allocate_block<'a, T>(
    size: usize,
    memspace: MemorySpace,
) -> Result<(*mut T, Layout), ViewError<'a>> {
    // this is a required check to use the layout obtained below in the alloc()
    assert_ne!(size, 0);
    let res = Layout::array::<T>(size);
    match res {
        Ok(layout) => {
            // the idea here is that we can use one allocator or the other
            // according to the specified memory space.
            // the Allocator trait is not yet stabilized though, so no
            // device-specific allocators are implementable atm (1.74.1)
            match memspace {
                MemorySpace::CPU => {
                    let ptr = unsafe { alloc(layout) };
                    if ptr.is_null() {
                        return Err(ViewError::Allocation(
                            "OOM / mismatch allocator's size or alignment constraints",
                        ));
                    }
                    Ok((ptr as *mut T, layout)) // need to return both for deallocation later
                }
                MemorySpace::GPU => {
                    unimplemented!()
                }
            }
        }
        Err(_) => {
            // this happens "on arithmetic overflow"
            // or "when the total size would exceed isize::MAX"
            // according to std::alloc::Layout::array documentation
            Err(ViewError::Allocation(
                "arithmetic overflow / total size > isize::MAX",
            ))
        }
    }
}

// ~~~~~~~~~ Tests ~~~~~~~~~

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_right() {
        // dim = [n0, n1, n2, n3]
        let dim = [3, 4, 5, 6];

        let cmp_stride = compute_stride(&dim, &MemoryLayout::Right);
        // n3 * n2 * n1, n3 * n2, n3, 1
        let ref_stride: [usize; 4] = [6 * 5 * 4, 6 * 5, 6, 1];

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn stride_left() {
        // dim = [n0, n1, n2, n3]
        let dim = [3, 4, 5, 6];

        let cmp_stride = compute_stride(&dim, &MemoryLayout::Left);
        // 1, n0, n0 * n1, n0 * n1 * n2
        let ref_stride: [usize; 4] = [1, 3, 3 * 4, 3 * 4 * 5];

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn one_d_stride() {
        // 1d view (vector) of length 1
        let dim: [usize; 1] = [8];
        let ref_stride: [usize; 1] = [1];
        let mut cmp_stride = compute_stride(&dim, &MemoryLayout::Right);
        assert_eq!(ref_stride, cmp_stride);
        cmp_stride = compute_stride(&dim, &MemoryLayout::Left);
        assert_eq!(ref_stride, cmp_stride);
    }
}
