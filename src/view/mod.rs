//!
//!
//!
//!

use std::sync::Arc;

use crate::traits::{
    data::SomeData,
    dimension::{Dim, Stride},
};

/// Maximum possible depth (i.e. number of dimensions) for a view.
pub const MAX_VIEW_DEPTH: usize = 8;

/// Enum used to represent data layout. Struct enums is used in order to increase
/// readability.
pub enum Layout<S>
where
    S: Stride,
{
    /// Highest stride for the first index, decreasing stride as index increases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    Right,
    /// Lowest stride for the first index, increasing stride as index decreases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    Left,
    /// Custom stride for each index. Must be compatible with dimensions.
    Stride { s: S },
}

/// Common structure used as the backend of all `View` types. The main differences between
/// usable types is the type of the `data` field.
pub struct ViewBase<T, S, D>
where
    T: SomeData,
    S: Stride,
    D: Dim,
{
    /// Data container. Depending on the type, it can be a vector (`Owned`), a reference
    /// (`ReadOnly`), a mutable reference (`ReadWrite`) or an `Arc` pointing on a vector
    /// (`Shared`).
    pub data: T,
    /// Memory layout of the view. Refer to Kokkos documentation for more information.
    pub layout: Layout<S>,
    /// Dimensions of the data represented by the view. The view can:
    /// - be a vector (1 dimension)
    /// - be a multi-dimensionnal array (up to 8 dimensions)
    /// The number of dimensions is referred to as the _depth_. Dimension 0, i.e. scalar,
    /// is not directly supported.
    pub dim: D,
    /// Stride between each element of a given dimension. Computed automatically for
    /// [Layout::Left] and [Layout::Right].
    pub stride: S,
}

impl<T, S, D> ViewBase<T, S, D>
where
    T: SomeData,
    S: Stride,
    D: Dim,
{
    /// Constructor.
    pub fn new(data: T, layout: Layout<S>, dim: D) -> Self {
        todo!()
    }
}

/// View type owning the data it yields access to, i.e. "original" view.
pub type ViewOwned<A, S, D> = ViewBase<Vec<A>, S, D>;

/// View type owning a read-only borrow to the data it yields access to, i.e. a
/// read-only mirror.
pub type ViewRO<'a, A, S, D> = ViewBase<&'a [A], S, D>;

/// View type owning a mutable borrow to the data it yields access to, i.e. a
/// read-write mirror.
pub type ViewRW<'a, A, S, D> = ViewBase<&'a mut [A], S, D>;

/// View type owning a shared reference to the data it yields access to, i.e. a
/// thread-safe read-only mirror.
pub type ViewShared<A, S, D> = ViewBase<Arc<Vec<A>>, S, D>;
