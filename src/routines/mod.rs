//! parallel statement related code
//!
//! This module contains code used for the implementations of parallel statements, e.g.
//! `parallel_for`, a Kokkos specific implementation of the commonly used pattern.
//!
//! `parallel_for` is currently the only statement considered for implementation;
//!
//! Parameters of aforementionned statements are defined in the [`parameters`] sub-module.
//!

pub mod dispatch;
pub mod parameters;

use std::fmt::Display;

use self::{
    dispatch::DispatchError,
    parameters::{ExecutionPolicy, RangePolicy},
};

// Enums

#[derive(Debug)]
pub enum StatementError {
    InconsistentDepth,
    Dispatch(DispatchError),
}

impl From<DispatchError> for StatementError {
    fn from(e: DispatchError) -> Self {
        StatementError::Dispatch(e)
    }
}

impl Display for StatementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatementError::InconsistentDepth => {
                write!(f, "inconsistent depth & range policy association")
            }
            StatementError::Dispatch(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for StatementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StatementError::InconsistentDepth => None,
            StatementError::Dispatch(e) => Some(e),
        }
    }
}

// Statements

pub fn parallel_for<const DEPTH: u8, const N: usize, F>(
    execp: ExecutionPolicy<N>,
    func: F,
) -> Result<(), StatementError>
// potentially a handle in the result if we can make the kernel execution async
where
    F: FnMut([usize; N]), // for statement should not return a result
{
    // checks...
    // hierarchy check
    let d: u8 = match execp.range {
        RangePolicy::RangePolicy(_) => 0,
        RangePolicy::MDRangePolicy(_) => 0,
        RangePolicy::TeamPolicy {
            league_size: _,
            team_size: _,
            vector_size: _,
        } => 0,
        RangePolicy::PerTeam => 1,
        RangePolicy::PerThread => 1,
        RangePolicy::TeamThreadRange => 1,
        RangePolicy::TeamThreadMDRange => 1,
        RangePolicy::TeamVectorRange => 1,
        RangePolicy::TeamVectorMDRange => 1,
        RangePolicy::ThreadVectorRange => 2,
        RangePolicy::ThreadVectorMDRange => 2,
    };
    assert_eq!(d, DEPTH);

    // data prep?

    // dispatch
    let res = match execp.space {
        parameters::ExecutionSpace::Serial => dispatch::serial(execp, func),
        parameters::ExecutionSpace::DeviceCPU => dispatch::cpu(),
        parameters::ExecutionSpace::DeviceGPU => dispatch::gpu(),
    };

    // Ok or converts error
    res.map_err(|e| e.into())
}
