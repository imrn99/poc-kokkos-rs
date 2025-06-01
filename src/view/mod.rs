//! data structure related code
//!
//! This module contains code used for the implementations of `Views`, a data structure
//! defined and used by the Kokkos library. There are different types of views, all
//! implemented using the same backend, [ViewBase].
//!
//! Eventually, the different types of Views should be removed and replaced by a single
//! type. The distinction between original and mirrors doesn't seem necessary in a Rust
//! implementation where the ownership system handles all memory transaction.
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
//!     parameters::Layout,
//!     ViewOwned,
//! };
//!
//! let mut viewA: ViewOwned<'_, 2, f64> = ViewOwned::new(
//!         Layout::Right, // see parameters & Kokkos doc
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

use self::parameters::{compute_stride, DataTraits, DataType, InnerDataType, Layout};
use std::{fmt::Debug, ops::Index};

#[derive(Debug)]
/// Enum used to classify view-related errors.
///
/// In all variants, the internal value is a description of the error.
pub enum ViewError<'a> {
    ValueError(&'a str),
    DoubleMirroring(&'a str),
}

#[derive(Debug, PartialEq)]
/// Common structure used as the backend of all `View` types. The main differences between
/// usable types is the type of the `data` field.
pub struct ViewBase<'a, const N: usize, T>
where
    T: DataTraits,
{
    /// Data container. Depending on the type, it can be a vector (`Owned`), a reference
    /// (`ReadOnly`) or a mutable reference (`ReadWrite`).
    pub data: DataType<'a, T>,
    /// Memory layout of the view. Refer to Kokkos documentation for more information.
    pub layout: Layout<N>,
    /// Dimensions of the data represented by the view. The view can:
    ///
    /// - be a vector (1 dimension)
    /// - be a multi-dimensionnal array (up to 8 dimensions)
    ///
    /// The number of dimensions is referred to as the _depth_. Dimension 0, i.e. scalar,
    /// is not directly supported.
    pub dim: [usize; N],
    /// Stride between each element of a given dimension. Computed automatically for
    /// [Layout::Left] and [Layout::Right].
    pub stride: [usize; N],
}

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
// ~~~~~~~~ Constructors
impl<const N: usize, T> ViewBase<'_, N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new(layout: Layout<N>, dim: [usize; N]) -> Self {
        // compute stride & capacity
        let stride = compute_stride(&dim, &layout);
        let capacity: usize = dim.iter().product();

        // build & return
        Self {
            data: DataType::Owned(vec![T::default(); capacity]), // should this be allocated though?
            layout,
            dim,
            stride,
        }
    }

    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new_from_data(data: Vec<T>, layout: Layout<N>, dim: [usize; N]) -> Self {
        // compute stride if necessary
        let stride = compute_stride(&dim, &layout);

        // checks
        let capacity: usize = dim.iter().product();
        assert_eq!(capacity, data.len());

        // build & return
        Self {
            data: DataType::Owned(data),
            layout,
            dim,
            stride,
        }
    }
}

#[cfg(any(feature = "rayon", feature = "threads", feature = "gpu"))]
// ~~~~~~~~ Constructors
impl<const N: usize, T> ViewBase<'_, N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new(layout: Layout<N>, dim: [usize; N]) -> Self {
        // compute stride & capacity
        let stride = compute_stride(&dim, &layout);
        let capacity: usize = dim.iter().product();

        // build & return
        Self {
            data: DataType::Owned((0..capacity).map(|_| Atomic::new(T::default())).collect()), // should this be allocated though?
            layout,
            dim,
            stride,
        }
    }

    /// Constructor used to create owned views. See dedicated methods for others.
    pub fn new_from_data(data: Vec<T>, layout: Layout<N>, dim: [usize; N]) -> Self {
        // compute stride if necessary
        let stride = compute_stride(&dim, &layout);

        // checks
        let capacity: usize = dim.iter().product();
        assert_eq!(capacity, data.len());

        // build & return
        Self {
            data: DataType::Owned(data.into_iter().map(|elem| Atomic::new(elem)).collect()),
            layout,
            dim,
            stride,
        }
    }
}

// ~~~~~~~~ Uniform writing interface across all features
impl<'a, const N: usize, T> ViewBase<'a, N, T>
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
    ///   regular [Index] trait implementation to prevent a mutable borrow. The store
    ///   currently uses relaxed ordering, this may change.
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
    ///   regular [Index] trait implementation to prevent a mutable borrow. The store
    ///   currently uses relaxed ordering, this may change.
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
    ///   regular [Index] trait implementation. The load currently uses relaxed ordering,
    ///   this may change.
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
    ///   regular [Index] trait implementation. The load currently uses relaxed ordering,
    ///   this may change.
    /// - no feature enabled: uses the regular [Index] trait implementation.
    ///
    /// Note that [Index] is always implemented while [IndexMut] only is when no
    /// features are enabled.
    ///
    /// **Current version**: thread-safe
    pub fn get(&self, index: [usize; N]) -> T {
        self[index].load(atomic::Ordering::Relaxed)
    }

    // ~~~~~~~~ Mirrors

    /// Create a new View mirroring `self`, i.e. referencing the same data. This mirror
    /// is always immutable, but it inner values might still be writable if they are
    /// atomic types.
    ///
    /// Note that mirrors currently can only be created from the "original" view,
    /// i.e. the view owning the data.
    pub fn create_mirror<'b>(&'a self) -> Result<ViewRO<'b, N, T>, ViewError<'a>>
    where
        'a: 'b, // 'a outlives 'b
    {
        let inner = if let DataType::Owned(v) = &self.data {
            v
        } else {
            return Err(ViewError::DoubleMirroring(
                "Cannot create a mirror from a non-data-owning View",
            ));
        };

        Ok(Self {
            data: DataType::Borrowed(inner),
            layout: self.layout,
            dim: self.dim,
            stride: self.stride,
        })
    }

    #[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Create a new View mirroring `self`, i.e. referencing the same data. This mirror
    /// uses a mutable reference, hence the serial-only definition
    ///
    /// Note that mirrors currently can only be created from the "original" view,
    /// i.e. the view owning the data.
    ///
    /// Only defined when no feature are enabled since all interfaces should be immutable
    /// otherwise.
    pub fn create_mutable_mirror<'b>(&'a mut self) -> Result<ViewRW<'b, N, T>, ViewError<'a>>
    where
        'a: 'b, // 'a outlives 'b
    {
        let inner = if let DataType::Owned(v) = &mut self.data {
            v
        } else {
            return Err(ViewError::DoubleMirroring(
                "Cannot create a mirror from a non-data-owning View",
            ));
        };

        Ok(Self {
            data: DataType::MutBorrowed(inner),
            layout: self.layout,
            dim: self.dim,
            stride: self.stride,
        })
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
        if let DataType::Owned(v) = self.data {
            Ok(v)
        } else {
            Err(ViewError::ValueError(
                "Cannot fetch raw values of a non-data-owning views",
            ))
        }
    }

    #[cfg(all(test, any(feature = "rayon", feature = "threads", feature = "gpu")))]
    /// Consumes the view to return a `Vec` containing its raw data content.
    ///
    /// This method is meant to be used in tests
    pub fn raw_val<'b>(self) -> Result<Vec<T>, ViewError<'b>> {
        if let DataType::Owned(v) = self.data {
            Ok(v.iter()
                .map(|elem| elem.load(atomic::Ordering::Relaxed))
                .collect::<Vec<T>>())
        } else {
            Err(ViewError::ValueError(
                "Cannot fetch raw values of a non-data-owning views",
            ))
        }
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

/// **Read-only access is always implemented.**
impl<const N: usize, T> Index<[usize; N]> for ViewBase<'_, N, T>
where
    T: DataTraits,
{
    type Output = InnerDataType<T>;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let flat_idx: usize = self.flat_idx(index);
        match &self.data {
            DataType::Owned(v) => {
                assert!(flat_idx < v.len()); // remove bounds check
                &v[flat_idx]
            }
            DataType::Borrowed(slice) => {
                assert!(flat_idx < slice.len()); // remove bounds check
                &slice[flat_idx]
            }
            DataType::MutBorrowed(mut_slice) => {
                assert!(flat_idx < mut_slice.len()); // remove bounds check
                &mut_slice[flat_idx]
            }
        }
    }
}

#[cfg(not(any(feature = "rayon", feature = "threads", feature = "gpu")))]
/// **Read-write access is only implemented when no parallel features are enabled.**
impl<const N: usize, T> IndexMut<[usize; N]> for ViewBase<'_, N, T>
where
    T: DataTraits,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let flat_idx: usize = self.flat_idx(index);
        match &mut self.data {
            DataType::Owned(v) => {
                assert!(flat_idx < v.len()); // remove bounds check
                &mut v[flat_idx]
            }
            DataType::Borrowed(_) => unimplemented!("Cannot mutably access a read-only view!"),
            DataType::MutBorrowed(mut_slice) => {
                assert!(flat_idx < mut_slice.len()); // remove bounds check
                &mut mut_slice[flat_idx]
            }
        }
    }
}

/// View type owning the data it yields access to, i.e. "original" view.
pub type ViewOwned<'a, const N: usize, T> = ViewBase<'a, N, T>;

/// View type owning a read-only borrow to the data it yields access to, i.e. a
/// read-only mirror.
pub type ViewRO<'a, const N: usize, T> = ViewBase<'a, N, T>;

/// View type owning a mutable borrow to the data it yields access to, i.e. a
/// read-write mirror.
pub type ViewRW<'a, const N: usize, T> = ViewBase<'a, N, T>;
