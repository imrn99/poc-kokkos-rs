//! parallel statement related code
//!
//! This module contains code used for the implementations of parallel statements, e.g.
//! `parallel_for`, a Kokkos specific implementation of the commonly used pattern.
//!
//! `parallel_for` is currently the only statement considered for implementation;
//!
//! Parameters of aforementionned statements are defined in the [`parameters`] sub-module.
//!

use self::parameters::{ExecutionPolicy, RangePolicy};

pub mod dispatch;
pub mod parameters;

pub fn parallel_for<const DEPTH: usize, R, F, E, Args, Error>(
    execp: ExecutionPolicy<R>,
    func: F,
) -> Result<(), E>
// potentially a handle in the reuslt if we can make the kernel execution async
where
    R: RangePolicy,
    F: FnMut(Args), // for statement should not return a result?
    E: std::error::Error,
{
    // checks...

    // data prep

    // dispatch
    todo!()
}
