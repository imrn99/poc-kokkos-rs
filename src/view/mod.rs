//!
//!
//!
//!

use std::sync::Arc;

use crate::traits::{
    data::SomeData,
    dimension::{Dim, Stride},
};

pub const MAX_VIEW_DEPTH: usize = 8;

/// Enum used to represent data layout. Struct enums is used in order to increase
/// readability.
pub enum Layout<S>
where
    S: Stride,
{
    /// Highest stride for the first index, decreasing stride as index increases.
    Right { s: S },
    /// Lowest stride for the first index, increasing stride as index decreases.
    Left { s: S },
    /// Custom stride for each index.
    Stride { s: S },
}

pub struct ViewBase<T, S, D>
where
    T: SomeData,
    S: Stride,
    D: Dim,
{
    pub data: T,
    pub layout: Layout<S>,
    pub dim: D,
}

pub type ViewOwned<A, S, D> = ViewBase<Vec<A>, S, D>;

pub type ViewRO<'a, A, S, D> = ViewBase<&'a A, S, D>;

pub type ViewRW<'a, A, S, D> = ViewBase<&'a mut A, S, D>;

pub type ViewShared<A, S, D> = ViewBase<Arc<Vec<A>>, S, D>;
