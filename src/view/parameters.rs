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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_right() {
        let mut dim: Dimension = smallvec!();
        dim.push(3); // n0
        dim.push(4); // n1
        dim.push(5); // n2
        dim.push(6); // n3

        let cmp_stride = compute_stride(&dim, &Layout::Right);
        let mut ref_stride: Stride = smallvec!();
        ref_stride.push(6 * 5 * 4); // n3 * n2 * n1
        ref_stride.push(6 * 5); // n3 * n2
        ref_stride.push(6); // n3
        ref_stride.push(1); // 1

        assert_eq!(cmp_stride, ref_stride);
    }

    #[test]
    fn stride_left() {
        let mut dim: Dimension = smallvec!();
        dim.push(3); // n0
        dim.push(4); // n1
        dim.push(5); // n2
        dim.push(6); // n3

        let cmp_stride = compute_stride(&dim, &Layout::Left);
        let mut ref_stride: Stride = smallvec!();
        ref_stride.push(1); // 1
        ref_stride.push(3); // n0
        ref_stride.push(3 * 4); // n0 * n1
        ref_stride.push(3 * 4 * 5); // n0 * n1 * n2

        assert_eq!(cmp_stride, ref_stride);
    }
}
