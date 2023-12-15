//! parallel statement related code
//!
//! This module contains code used for the implementation of parallel statements, e.g.
//! `parallel_for`, a Kokkos specific implementation of commonly used patterns.
//!
//! Parameters of aforementionned statements are defined in the [`parameters`] sub-module.
//!
//! Dispatch code is defined in the [`dispatch`] sub-module.
//!
//! Currently implemented statements:
//!
//! - `parallel_for`

pub mod dispatch;
pub mod parameters;

use std::fmt::Display;

use crate::functor::KernelArgs;

use self::{dispatch::DispatchError, parameters::ExecutionPolicy};

// Enums

/// Enum used to classify possible errors occuring in a parallel statement.
#[derive(Debug)]
pub enum StatementError {
    /// Error occured during dispatch; The specific [DispatchError] is
    /// used as the internal value of this variant.
    Dispatch(DispatchError),
    /// Error raised when parallel hierarchy isn't respected.
    InconsistentDepth,
    /// What did I mean by this?
    InconsistentExecSpace,
}

impl From<DispatchError> for StatementError {
    fn from(e: DispatchError) -> Self {
        StatementError::Dispatch(e)
    }
}

impl Display for StatementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatementError::Dispatch(e) => write!(f, "{}", e),
            StatementError::InconsistentDepth => {
                write!(f, "inconsistent depth & range policy association")
            }
            StatementError::InconsistentExecSpace => {
                write!(f, "?")
            }
        }
    }
}

impl std::error::Error for StatementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StatementError::Dispatch(e) => Some(e),
            StatementError::InconsistentDepth => None,
            StatementError::InconsistentExecSpace => None,
        }
    }
}

// Statements

// All of this would be half as long if impl trait in type aliases was stabilized

cfg_if::cfg_if! {
    if #[cfg(feature = "threads")] {
        /// Parallel For statement.
        ///
        /// Current version: `threads`
        pub fn parallel_for<const N: usize>(
            execp: ExecutionPolicy<N>,
            func: impl Fn(KernelArgs<N>) + Send + Sync + Clone,
        ) -> Result<(), StatementError> {
            // checks...

            // data prep?
            let kernel = Box::new(func);

            // dispatch
            let res = match execp.space {
                parameters::ExecutionSpace::Serial => dispatch::serial(execp, kernel),
                parameters::ExecutionSpace::DeviceCPU => dispatch::cpu(execp, kernel),
                parameters::ExecutionSpace::DeviceGPU => dispatch::gpu(execp, kernel),
            };

            // Ok or converts error
            res.map_err(|e| e.into())
        }
    } else if #[cfg(feature = "rayon")] {
        /// Parallel For statement.
        ///
        /// Current version: `rayon`
        pub fn parallel_for<const N: usize>(
            execp: ExecutionPolicy<N>,
            func: impl Fn(KernelArgs<N>) + Send + Sync,
        ) -> Result<(), StatementError> {
            // checks...

            // data prep?
            let kernel = Box::new(func);

            // dispatch
            let res = match execp.space {
                parameters::ExecutionSpace::Serial => dispatch::serial(execp, kernel),
                parameters::ExecutionSpace::DeviceCPU => dispatch::cpu(execp, kernel),
                parameters::ExecutionSpace::DeviceGPU => dispatch::gpu(execp, kernel),
            };

            // Ok or converts error
            res.map_err(|e| e.into())
        }
    } else {
        /// Parallel For statement.
        ///
        /// Current version: no feature
        pub fn parallel_for<const N: usize>(
            execp: ExecutionPolicy<N>,
            func: impl FnMut(KernelArgs<N>),
        ) -> Result<(), StatementError> {
            // checks...

            // data prep?
            let kernel = Box::new(func);

            // dispatch
            let res = match execp.space {
                parameters::ExecutionSpace::Serial => dispatch::serial(execp, kernel),
                parameters::ExecutionSpace::DeviceCPU => dispatch::cpu(execp, kernel),
                parameters::ExecutionSpace::DeviceGPU => dispatch::gpu(execp, kernel),
            };

            // Ok or converts error
            res.map_err(|e| e.into())
        }
    }
}
