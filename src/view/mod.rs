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
//! Parameters of aforementionned views are defined in the [`parameters`] sub-module.
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

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
use atomic::{Atomic, Ordering};

#[cfg(any(doc, not(any(feature = "rayon", feature = "threads", feature = "gpu"))))]
use std::ops::IndexMut;

use self::parameters::{compute_stride, DataTraits, InnerDataType, MemoryLayout};
use std::{fmt::Debug, ops::Index};

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
    /// Data container. Depending on the type, it can be a vector (`Owned`), a reference
    /// (`ReadOnly`) or a mutable reference (`ReadWrite`).
    pub data: Vec<InnerDataType<T>>,
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

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
// ~~~~~~~~ Constructors
impl<const N: usize, T> View<N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new(layout: MemoryLayout<N>, dim: [usize; N]) -> Self {
        // compute stride & capacity
        let stride = compute_stride(&dim, &layout);
        let capacity: usize = dim.iter().product();

        // build & return
        Self {
            data: vec![T::default(); capacity], // should this be allocated though?
            layout,
            dim,
            stride,
        }
    }

    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new_from_data(data: Vec<T>, layout: MemoryLayout<N>, dim: [usize; N]) -> Self {
        // compute stride if necessary
        let stride = compute_stride(&dim, &layout);

        // checks
        let capacity: usize = dim.iter().product();
        assert_eq!(capacity, data.len());

        // build & return
        Self {
            data,
            layout,
            dim,
            stride,
        }
    }
}

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
// ~~~~~~~~ Constructors
impl<const N: usize, T> View<N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new(layout: MemoryLayout<N>, dim: [usize; N]) -> Self {
        // compute stride & capacity
        let stride = compute_stride(&dim, &layout);
        let capacity: usize = dim.iter().product();

        // build & return
        Self {
            data: (0..capacity).map(|_| Atomic::new(T::default())).collect(),
            layout,
            dim,
            stride,
        }
    }

    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new_from_data(data: Vec<T>, layout: MemoryLayout<N>, dim: [usize; N]) -> Self {
        // compute stride if necessary
        let stride = compute_stride(&dim, &layout);

        // checks
        let capacity: usize = dim.iter().product();
        assert_eq!(capacity, data.len());

        // build & return
        Self {
            data: data.into_iter().map(|elem| Atomic::new(elem)).collect(),
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
    /// Two different implementations of this method are defined in order to satisfy
    /// the (im)mutability requirements when using parallelization features & keep a
    /// consistent user API:
    ///
    /// - any feature enabled: implictly use an atomic store operation on top of the
    /// regular [Index] trait implementation to prevent a mutable borrow. The store
    /// currently uses relaxed ordering, this may change.
    /// - no feature enabled: uses a regular [IndexMut] trait implementation.
    ///
    /// Note that [Index] is always implemented while [IndexMut] only is when no
    /// features are enabled.
    ///
    /// **Current version**: no feature
    pub fn set(&mut self, index: [usize; N], val: T) {
        self[index] = val;
    }

    #[inline(always)]
    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Writing interface.
    ///
    /// Two different implementations of this method are defined in order to satisfy
    /// the (im)mutability requirements when using parallelization features & keep a
    /// consistent user API:
    ///
    /// - any feature enabled: implictly use an atomic store operation on top of the
    /// regular [Index] trait implementation to prevent a mutable borrow. The store
    /// currently uses relaxed ordering, this may change.
    /// - no feature enabled: uses a regular [IndexMut] trait implementation.
    ///
    /// Note that [Index] is always implemented while [IndexMut] only is when no
    /// features are enabled.
    ///
    /// **Current version**: thread-safe
    pub fn set(&self, index: [usize; N], val: T) {
        self[index].store(val, Ordering::Relaxed);
    }

    #[inline(always)]
    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Reading interface.
    ///
    /// Two different implementations of this method are defined in order to keep a
    /// consistent user API across features:
    ///
    /// - any feature enabled: implictly use an atomic load operation on top of the
    /// regular [Index] trait implementation. The load currently uses relaxed ordering,
    /// this may change.
    /// - no feature enabled: uses the regular [Index] trait implementation.
    ///
    /// Note that [Index] is always implemented while [IndexMut] only is when no
    /// features are enabled.
    ///
    /// **Current version**: no feature
    pub fn get(&self, index: [usize; N]) -> T {
        self[index]
    }

    #[inline(always)]
    #[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
    /// Reading interface.
    ///
    /// Two different implementations of this method are defined in order to keep a
    /// consistent user API across features:
    ///
    /// - any feature enabled: implictly use an atomic load operation on top of the
    /// regular [Index] trait implementation. The load currently uses relaxed ordering,
    /// this may change.
    /// - no feature enabled: uses the regular [Index] trait implementation.
    ///
    /// Note that [Index] is always implemented while [IndexMut] only is when no
    /// features are enabled.
    ///
    /// **Current version**: thread-safe
    pub fn get(&self, index: [usize; N]) -> T {
        self[index].load(atomic::Ordering::Relaxed)
    }

    // ~~~~~~~~ Convenience

    #[cfg(all(
        test,
        not(any(feature = "rayon", feature = "threads", feature = "gpu"))
    ))]
    /// Consumes the view to return a `Vec` containing its raw data content.
    ///
    /// This method is meant to be used in tests
    pub fn raw_val<'b>(self) -> Result<Vec<T>, ViewError<'b>> {
        Ok(self.data)
    }

    #[cfg(all(test, any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Consumes the view to return a `Vec` containing its raw data content.
    ///
    /// This method is meant to be used in tests
    pub fn raw_val<'b>(self) -> Result<Vec<T>, ViewError<'b>> {
        Ok(self
            .data
            .iter()
            .map(|elem| elem.load(atomic::Ordering::Relaxed))
            .collect::<Vec<T>>())
    }

    #[inline(always)]
    /// Mapping function between N-indices and the flat offset.
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
        self.data.as_ptr() == other.data.as_ptr()
        // meta data just needs strict equality
            && self.layout == other.layout
            && self.dim == other.dim
            && self.stride == other.stride
    }
}

/// **Read-only access is always implemented.**
impl<const N: usize, T> Index<[usize; N]> for View<N, T>
where
    T: DataTraits,
{
    type Output = InnerDataType<T>;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let flat_idx: usize = self.flat_idx(index);
        assert!(flat_idx < self.data.len()); // remove bounds check
        &self.data[flat_idx]
    }
}

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
/// **Read-write access is only implemented when no parallel features are enabled.**
impl<const N: usize, T> IndexMut<[usize; N]> for View<N, T>
where
    T: DataTraits,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let flat_idx: usize = self.flat_idx(index);
        assert!(flat_idx < self.data.len()); // remove bounds check
        &mut self.data[flat_idx]
    }
}
