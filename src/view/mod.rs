//! data structure related code
//!
//! This module contains code used for the implementations of *Views*, a data structure
//! defined and used by the Kokkos library.
//!
//! In order to have thread-safe structures to use in parallel statement, the inner data
//! type of views is adjusted implicitly when compiling using parallelization features.
//! To match the adjusted data type, view access is done through `get` and `set` methods,
//! allowing for feature-specific mutability in signatures while keeping a consistent user
//! API.
//!
//! Parameters of aforementionned views are defined in the [`parameters`] sub-module. The
//! module also, contains the internal types used to implement views.
//!
//! ### Example
//!
//! Initialize and fill a 2D matrix:
//! ```rust
//! use poc_kokkos_rs::view::{
//!     parameters::MemoryLayout,
//!     View,
//! };
//!
//! let mut viewA: View<2, f64> = View::new(
//!         MemoryLayout::Right, // see parameters & Kokkos doc
//!         [3, 5],        // 3 rows, 5 columns
//!     );
//!
//! for row in 0..3 {
//!     for col in 0..5 {
//!         viewA.set([row, col], row as f64);
//!     }
//! }
//!
//! // viewA:
//! // (0.0 0.0 0.0 0.0 0.0)
//! // (1.0 1.0 1.0 1.0 1.0)
//! // (2.0 2.0 2.0 2.0 2.0)
//! ```

pub mod parameters;

use self::parameters::{compute_stride, DataTraits, MemoryLayout, MemorySpace, ViewData};
use std::fmt::Debug;

#[derive(Debug)]
/// Enum used to classify view-related errors.
///
/// In all variants, the internal value is a description of the error.
pub enum ViewError<'a> {
    Allocation(&'a str),
    ValueError(&'a str),
    DoubleMirroring(&'a str),
}

#[derive(Debug)]
/// Main View structure.
///
/// A `View` represent an N-dimensionnal array of type T. The implementation uses a
/// const generic to handle dimension-specific operations.
pub struct View<const N: usize, T>
where
    T: DataTraits,
{
    /// Data container. This is a custom structure, roughly equivalent to a
    /// stripped-down vector. Most technical operations are defined through this
    /// structure rather than through views directly.
    pub data: ViewData<T>,
    /// Memory layout of the view. Refer to Kokkos documentation for more information.
    pub layout: MemoryLayout<N>,
    /// Dimensions of the data represented by the view. The view can:
    /// - be a vector (1 dimension)
    /// - be a multi-dimensionnal array (up to 8 dimensions)
    /// The number of dimensions is referred to as the _depth_. Dimension 0, i.e. scalar,
    /// is not directly supported at the moment.
    pub dim: [usize; N],
    /// Stride between each element of a given dimension. Computed automatically for
    /// [MemoryLayout::Left] and [MemoryLayout::Right].
    pub stride: [usize; N],
}

// ~~~~~~~~ Constructors
impl<const N: usize, T> View<N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor.
    pub fn new(layout: MemoryLayout<N>, dim: [usize; N]) -> Self {
        // compute stride & capacity
        let stride = compute_stride(&dim, &layout);
        let capacity: usize = dim.iter().product();

        // build data holder
        let mut viewdata: ViewData<T> = ViewData::new(capacity, MemorySpace::CPU);
        viewdata.fill(T::default());

        // build & return
        Self {
            data: viewdata,
            layout,
            dim,
            stride,
        }
    }

    /// Constructor.
    pub fn new_from_data(data: Vec<T>, layout: MemoryLayout<N>, dim: [usize; N]) -> Self {
        // compute stride if necessary
        let stride = compute_stride(&dim, &layout);
        let capacity: usize = dim.iter().product();

        let mut viewdata: ViewData<T> = ViewData::new(capacity, MemorySpace::CPU);
        viewdata.take_vals(&data);

        // build & return
        Self {
            data: viewdata,
            layout,
            dim,
            stride,
        }
    }
}

// ~~~~~~~~ Uniform writing interface across all features
impl<const N: usize, T> View<N, T>
where
    T: DataTraits,
{
    #[inline(always)]
    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Writing interface.
    ///
    /// Passthrough to the equivalent [ViewData][ViewData::set] method.
    ///
    /// **Current version**: no feature
    pub fn set(&mut self, index: [usize; N], val: T) {
        let flat_index = self.flat_idx(index);
        self.data.set(flat_index, val);
    }

    #[inline(always)]
    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Writing interface.
    ///
    /// Passthrough to the equivalent [ViewData][ViewData::set] method.
    ///
    /// **Current version**: thread-safe
    pub fn set(&self, index: [usize; N], val: T) {
        let flat_index = self.flat_idx(index);
        self.data.set(flat_index, val);
    }

    #[inline(always)]
    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Reading interface.
    ///
    /// Passthrough to the equivalent [ViewData][ViewData::get] method.
    ///
    /// **Current version**: no feature
    pub fn get(&self, index: [usize; N]) -> T {
        let flat_index = self.flat_idx(index);
        self.data.get(flat_index)
    }

    #[inline(always)]
    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Reading interface.
    ///
    /// Passthrough to the equivalent [ViewData][ViewData::get] method.
    ///
    /// **Current version**: thread-safe
    pub fn get(&self, index: [usize; N]) -> T {
        let flat_index = self.flat_idx(index);
        self.data.get(flat_index)
    }

    // ~~~~~~~~ Convenience

    #[cfg(all(
        test,
        not(any(feature = "rayon", feature = "threads", feature = "gpu"))
    ))]
    /// Consumes the view to return a `Vec` containing its raw data content.
    ///
    /// This method is meant to be used in tests
    pub fn raw_val(self) -> Vec<T> {
        (0..self.data.size)
            // the Deref should result in copied values, hence not pose any problem with the deallocation.
            .map(|idx| self.data.get(idx))
            .collect()
    }

    #[cfg(all(test, any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Consumes the view to return a `Vec` containing its raw data content.
    ///
    /// This method is meant to be used in tests
    pub fn raw_val(self) -> Vec<T> {
        (0..self.data.size)
            // the Deref should result in copied values, hence not pose any problem with the deallocation.
            .map(|idx| self.data.get(idx))
            .collect()
    }

    #[inline(always)]
    /// Mapping function between N-indices and the flat offset in the allocated memory.
    pub fn flat_idx(&self, index: [usize; N]) -> usize {
        index
            .iter()
            .zip(self.stride.iter())
            .map(|(i, s_i)| *i * *s_i)
            .sum()
    }
}

/// The policy used to implement the [PartialEq] trait is based on Kokkos'
/// [`equal` algorithm](https://kokkos.github.io/kokkos-core-wiki/API/algorithms/std-algorithms/all/StdEqual.html).
/// Essentially, it corresponds to equality by reference instead of equality by value.
impl<const N: usize, T> PartialEq for View<N, T>
where
    T: DataTraits,
{
    fn eq(&self, other: &Self) -> bool {
        // kokkos implements equality by reference
        // i.e. two views are equal if they reference
        // the same memory space.
        self.data.ptr_is_eq(&other.data)
        // meta data just needs strict equality
            && self.layout == other.layout
            && self.dim == other.dim
            && self.stride == other.stride
    }
}
