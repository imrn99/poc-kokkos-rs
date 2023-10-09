//!
//!
//!
//!

use smallvec::{smallvec, SmallVec};

/// Maximum possible depth (i.e. number of dimensions) for a view.
pub const MAX_VIEW_DEPTH: usize = 8;

/// Dimension
pub type Dimension = SmallVec<[usize; MAX_VIEW_DEPTH]>;

/// Stride
pub type Stride = SmallVec<[usize; MAX_VIEW_DEPTH]>;

#[derive(Clone)]
/// Enum used to represent data layout. Struct enums is used in order to increase
/// readability.
pub enum Layout {
    /// Highest stride for the first index, decreasing stride as index increases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    Right,
    /// Lowest stride for the first index, increasing stride as index decreases.
    /// Exact stride for each index can be computed from dimensions at view initialization.
    Left,
    /// Custom stride for each index. Must be compatible with dimensions.
    Stride { s: Stride },
}

pub fn compute_stride(dim: &Dimension, layout: &Layout) -> Stride {
    match layout.clone() {
        Layout::Right => {
            let mut stride: Stride = smallvec!();

            let mut tmp: usize = 1;
            stride.push(tmp);
            for i in (1..dim.len()).rev() {
                tmp *= dim[i];
                stride.push(tmp);
            }

            stride.reverse();
            stride
        }
        Layout::Left => {
            let mut stride: Stride = smallvec!();

            let mut tmp: usize = 1;
            stride.push(tmp);
            for i in 0..dim.len() - 1 {
                tmp *= dim[i];
                stride.push(tmp);
            }

            stride
        }
        Layout::Stride { s } => s,
    }
}
