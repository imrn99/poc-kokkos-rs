use std::ops::Index;

pub trait Dim: Index<usize> {
    fn depth(&self) -> usize;
    fn val(&self) -> &[usize];
}

//TODO: write a macro that generates these blocks

impl Dim for [usize; 1] {
    fn depth(&self) -> usize {
        self.len()
    }

    fn val(&self) -> &[usize] {
        self
    }
}

pub trait Stride: Copy + Index<usize> {
    fn depth(&self) -> usize;
    fn val(&self) -> &[usize];
}

//TODO: write a macro that generates these blocks

// huh?
impl Stride for [usize; 0] {
    fn depth(&self) -> usize {
        self.len()
    }
    // huh?
    fn val(&self) -> &[usize] {
        self
    }
}

impl Stride for [usize; 1] {
    fn depth(&self) -> usize {
        self.len()
    }

    fn val(&self) -> &[usize] {
        self
    }
}
