//!
//!
//!
//!

pub const MAX_VIEW_DEPTH: usize = 5;

pub enum Layout {
    Right,
    Left,
}

pub struct View<T> {
    pub data: Vec<T>,
    pub layout: Layout,
    //pub dim: ? small vec of size 5 max ?
}

impl<T> View<T> {
    pub fn new(data: Vec<T>, lyt: Layout /*, dim:*/) -> Self {
        todo!()
    }
}
