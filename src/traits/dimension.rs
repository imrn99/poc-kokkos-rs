use std::ops::Index;

// Dim

pub trait Dim: Index<usize> {
    fn depth(&self) -> usize;
    fn val(&self) -> &[usize];
}

macro_rules! impl_dim {
    ($n:expr) => {
        impl Dim for [usize; $n] {
            fn depth(&self) -> usize {
                self.len()
            }

            fn val(&self) -> &[usize] {
                self
            }
        }
    };
}

impl_dim!(1);
impl_dim!(2);
impl_dim!(3);
impl_dim!(4);
impl_dim!(5);
impl_dim!(6);
impl_dim!(7);
impl_dim!(8);

// Stride

pub trait Stride: Copy + Index<usize> {
    fn depth(&self) -> usize;
    fn val(&self) -> &[usize];
}

macro_rules! impl_stride {
    ($n:expr) => {
        impl Stride for [usize; $n] {
            fn depth(&self) -> usize {
                self.len()
            }

            fn val(&self) -> &[usize] {
                self
            }
        }
    };
}

impl_stride!(1);
impl_stride!(2);
impl_stride!(3);
impl_stride!(4);
impl_stride!(5);
impl_stride!(6);
impl_stride!(7);
impl_stride!(8);
