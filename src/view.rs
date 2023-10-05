//!
//!
//!
//!

pub const MAX_VIEW_DEPTH: usize = 8;

pub enum Layout {
    Right,
    Left,
}

pub struct View<T> {
    pub data: Vec<T>,
    pub layout: Layout,
    //pub dim: ? small vec of size 5 max ?
}
