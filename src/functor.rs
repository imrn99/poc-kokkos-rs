//!
//!
//!
//!

/// Functor trait. User can implement its own functor by implementing this trait.
pub trait Functor<Args> {
    type Output;

    // forbidden becuase of https://github.com/rust-lang/rust/issues/91611
    // fn closure(&self) -> (impl Fn(Args) -> Self::Output);
    fn closure<'a, 'b>(&'a self) -> Box<dyn Fn(Args) -> Self::Output + 'b>
    where
        'a: 'b;
}

impl<Args, Out> Functor<Args> for dyn Fn(Args) -> Out {
    type Output = Out;

    fn closure<'a, 'b>(&'a self) -> Box<dyn Fn(Args) -> Self::Output + 'b>
    where
        'a: 'b,
    {
        Box::new(self)
    }
}
