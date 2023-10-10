//! data structure related code
//!
//! This module contains code used for the implementations of `Views`, a data structure
//! defined and used by the Kokkos library. There are different types of views, all
//! implemented using the same backend, [ViewBase].
//!
//! Parameters of aforementionned views are defined in the [`parameters`] sub-module.
//!

pub mod parameters;

use self::parameters::{compute_stride, DataType, Layout};
use std::{
    ops::{Index, IndexMut},
    sync::Arc,
};

/// Common structure used as the backend of all `View` types. The main differences between
/// usable types is the type of the `data` field.
pub struct ViewBase<'a, const N: usize, T> {
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

impl<'a, const N: usize, T> ViewBase<'a, N, T>
where
    T: Default + Clone, // fair assumption imo
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
        let depth = dim.len();
        // compute stride if necessary
        let stride = compute_stride(&dim, &layout);

        // checks
        assert_eq!(depth, stride.len());

        // build & return
        Self {
            data: DataType::Owned(data),
            layout,
            dim,
            stride,
        }
    }

    pub fn create_mirror<'b>(&'a self) -> ViewRO<'b, N, T>
    where
        'a: 'b,
    {
        let DataType::Owned(inner) = &self.data else {
            todo!()
        };
        let data = DataType::Borrowed(inner);

        Self {
            data,
            layout: self.layout,
            dim: self.dim,
            stride: self.stride,
        }
    }

    pub fn create_mutable_mirror<'b>(&'a mut self) -> ViewRW<'b, N, T>
    where
        'a: 'b,
    {
        let DataType::Owned(inner) = &mut self.data else {
            todo!()
        };
        let data = DataType::MutBorrowed(inner);

        Self {
            data,
            layout: self.layout,
            dim: self.dim,
            stride: self.stride,
        }
    }
}

impl<'a, const N: usize, T> Index<[usize; N]> for ViewBase<'a, N, T> {
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let flat_idx: usize = index
            .iter()
            .zip(self.stride.iter())
            .map(|(i, s_i)| *i * *s_i)
            .sum();
        match &self.data {
            DataType::Owned(v) => &v[flat_idx],
            DataType::Borrowed(slice) => &slice[flat_idx],
            DataType::MutBorrowed(mut_slice) => &mut_slice[flat_idx],
        }
    }
}

impl<'a, const N: usize, T> IndexMut<[usize; N]> for ViewBase<'a, N, T> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let flat_idx: usize = index
            .iter()
            .zip(self.stride.iter())
            .map(|(i, s_i)| *i * *s_i)
            .sum();
        match &mut self.data {
            DataType::Owned(v) => &mut v[flat_idx],
            DataType::Borrowed(_) => unimplemented!("Cannot mutably access a read-only view!"),
            DataType::MutBorrowed(mut_slice) => &mut mut_slice[flat_idx],
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

/// View type owning a shared reference to the data it yields access to, i.e. a
/// thread-safe read-only mirror. Is this useful ? Shouldn't this be Arc<Mutex<T>> ?
pub type ViewShared<'a, const N: usize, T> = ViewBase<'a, N, Arc<T>>;
