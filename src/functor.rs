//!
//!
//!
//!

/// Functor trait. User can implement its own functor by implementing this trait.
pub trait Functor<Args, Output>: Fn(Args) -> Output {}
