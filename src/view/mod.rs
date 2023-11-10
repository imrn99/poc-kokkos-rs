//! data structure related code
//!
//! This module contains code used for the implementations of `Views`, a data structure
//! defined and used by the Kokkos library. There are different types of views, all
//! implemented using the same backend, [ViewBase].
//!
//! Parameters of aforementionned views are defined in the [`parameters`] sub-module.
//!

pub mod parameters;

#[cfg(any(feature = "rayon", feature = "threads"))]
use atomic::Atomic;

use self::parameters::{compute_stride, DataTraits, DataType, InnerDataType, Layout};
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    sync::Arc,
};

#[derive(Debug)]
/// Enum used to classify view-related errors.
///
/// In all variants, the internal value is a description of the error.
pub enum ViewError<'a> {
    ValueError(&'a str),
    DoubleMirroring(&'a str),
}

#[derive(Debug)]
/// Common structure used as the backend of all `View` types. The main differences between
/// usable types is the type of the `data` field.
pub struct ViewBase<'a, const N: usize, T>
where
    T: DataTraits,
{
    /// Data container. Depending on the type, it can be a vector (`Owned`), a reference
    /// (`ReadOnly`), a mutable reference (`ReadWrite`) or an `Arc<>` pointing on a vector
    /// (`Shared`).
    pub data: DataType<'a, T>,
    /// Memory layout of the view. Refer to Kokkos documentation for more information.
    pub layout: Layout<N>,
    /// Dimensions of the data represented by the view. The view can:
    /// - be a vector (1 dimension)
    /// - be a multi-dimensionnal array (up to 8 dimensions)
    /// The number of dimensions is referred to as the _depth_. Dimension 0, i.e. scalar,
    /// is not directly supported.
    pub dim: [usize; N],
    /// Stride between each element of a given dimension. Computed automatically for
    /// [Layout::Left] and [Layout::Right].
    pub stride: [usize; N],
}

#[cfg(not(any(feature = "rayon", feature = "threads")))]
// ~~~~~~~~ Constructors
impl<'a, const N: usize, T> ViewBase<'a, N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor used to create owned (and shared?) views. See dedicated methods for
    /// others.
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

    /// Constructor used to create owned (and shared?) views. See dedicated methods for
    /// others.
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

#[cfg(any(feature = "rayon", feature = "threads"))]
// ~~~~~~~~ Constructors
impl<'a, const N: usize, T> ViewBase<'a, N, T>
where
    T: DataTraits, // fair assumption imo
{
    /// Constructor used to create owned (and shared?) views. See dedicated methods for
    /// others.
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

    /// Constructor used to create owned (and shared?) views. See dedicated methods for
    /// others.
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

impl<'a, const N: usize, T> ViewBase<'a, N, T>
where
    T: DataTraits,
{
    // ~~~~~~~~ Uniform writing interface across all features

    #[inline(always)]
    #[cfg(not(any(feature = "rayon", feature = "threads")))]
    /// Serial writing interface. Uses mutable indexing implementation.
    pub fn set(&mut self, index: [usize; N], val: T) {
        self[index] = val;
    }

    #[inline(always)]
    #[cfg(any(feature = "rayon", feature = "threads"))]
    /// Thread-safe writing interface. Uses non-mutable indexing and
    /// immutability of atomic type methods.
    pub fn set(&self, index: [usize; N], val: T) {
        self[index].store(val, atomic::Ordering::Relaxed);
    }

    // ~~~~~~~~ Mirrors

    /// Create a new View mirroring `self`, i.e. referencing the same data. This mirror
    /// is always immutable, but it inner values might still be writable if they are
    /// atomic types.
    ///
    /// Note that mirrors currently can only be created from the "original" view,
    /// i.e. the view owning the data.
    pub fn create_mirror<'b>(&'a self) -> Result<ViewRO<'b, N, T>, ViewError>
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

    #[cfg(not(any(feature = "rayon", feature = "threads")))]
    /// Only defined when no feature are enabled since all interfaces should be immutable
    /// otherwise.
    ///
    /// Create a new View mirroring `self`, i.e. referencing the same data. This mirror
    /// uses a mutable reference, hence the serial-only definition
    ///
    /// Note that mirrors currently can only be created from the "original" view,
    /// i.e. the view owning the data.
    pub fn create_mutable_mirror<'b>(&'a mut self) -> Result<ViewRW<'b, N, T>, ViewError>
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

    pub fn raw_val<'b>(self) -> Result<Vec<InnerDataType<T>>, ViewError<'b>> {
        if let DataType::Owned(v) = self.data {
            Ok(v)
        } else {
            Err(ViewError::ValueError(
                "Cannot fetch raw values of a non-data-owning views",
            ))
        }
    }

    #[inline(always)]
    pub fn flat_idx(&self, index: [usize; N]) -> usize {
        index
            .iter()
            .zip(self.stride.iter())
            .map(|(i, s_i)| *i * *s_i)
            .sum()
    }
}

/// Read-only access is always implemented.
impl<'a, const N: usize, T> Index<[usize; N]> for ViewBase<'a, N, T>
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

#[cfg(not(any(feature = "rayon", feature = "threads")))]
/// Read-write access is implemented using [IndexMut] trait when no parallel
/// features are enabled.
impl<'a, const N: usize, T> IndexMut<[usize; N]> for ViewBase<'a, N, T>
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

impl<'a, const N: usize, T: PartialEq + Debug> PartialEq for ViewBase<'a, N, T>
where
    T: DataTraits,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.layout == other.layout
            && self.dim == other.dim
            && self.stride == other.stride
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

/// View type owning a shared reference to the data it yields access to, i.e. a
/// thread-safe read-only mirror. Is this useful ? Shouldn't this be `Arc<Mutex<T>>` ?
pub type ViewShared<'a, const N: usize, T> = ViewBase<'a, N, Arc<T>>;
