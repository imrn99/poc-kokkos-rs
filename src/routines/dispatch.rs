//!
//!
//!
//!

use std::{fmt::Display, ops::Range};

use super::parameters::{ExecutionPolicy, RangePolicy};

// enums

#[derive(Debug)]
pub enum DispatchError {
    Serial,
    CPU,
    GPU,
}

impl Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::Serial => write!(f, "error during serial dispatch"),
            DispatchError::CPU => write!(f, "error during cpu dispatch"),
            DispatchError::GPU => write!(f, "error during gpu dispatch"),
        }
    }
}

impl std::error::Error for DispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DispatchError::Serial => None,
            DispatchError::CPU => None,
            DispatchError::GPU => None,
        }
    }
}

// dispatch routines

// internl routines

fn recursive_loop<const N: usize, F>(ranges: &[Range<usize>; N], mut kernel: F)
where
    F: FnMut([usize; N]),
{
    // handles recursions
    fn inner<const N: usize, F>(
        current_depth: usize,
        ranges: &[Range<usize>; N],
        kernel: &mut F,
        indices: &mut [usize; N],
    ) where
        F: FnMut([usize; N]),
    {
        if current_depth == N {
            // all loops unraveled
            // call the kernel
            kernel(*indices)
        } else {
            // loop on next dimension; update indices
            ranges[current_depth].clone().for_each(|i_current| {
                indices[current_depth] = i_current;
                inner(current_depth + 1, ranges, kernel, indices);
            });
        }
    }

    let mut indices = [0; N];
    inner(0, ranges, &mut kernel, &mut indices);
}

// serial dispatch

pub fn serial<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
where
    F: FnMut([usize; N]), // FnMut is the closure trait taken by for_each method
{
    match execp.range {
        RangePolicy::RangePolicy(range) => {
            // serial, 1D range
            assert_eq!(N, 1);
            // making indices N-sized arrays is necessary, even with the assertion...
            range.into_iter().map(|i| [i; N]).for_each(kernel)
        }
        RangePolicy::MDRangePolicy(ranges) => {
            // macros would pbly be more efficient
            recursive_loop(&ranges, kernel)
        }
        RangePolicy::TeamPolicy {
            league_size, // number of teams, i.e. chunks
            team_size,   // size of teams, i.e. chunk size
            vector_size,
        } => todo!(),
        RangePolicy::PerTeam => todo!(),
        RangePolicy::PerThread => todo!(),
        RangePolicy::TeamThreadRange => todo!(),
        RangePolicy::TeamThreadMDRange => todo!(),
        RangePolicy::TeamVectorRange => todo!(),
        RangePolicy::TeamVectorMDRange => todo!(),
        RangePolicy::ThreadVectorRange => todo!(),
        RangePolicy::ThreadVectorMDRange => todo!(),
    };
    todo!()
}

cfg_if::cfg_if! {
    if #[cfg(threads)] {
        // OS threads backend
        pub fn cpu() -> Result<(), DispatchError> {
            todo!()
        }
    } else if #[cfg(rayon)] {
        // rayon backend
        pub fn cpu() -> Result<(), DispatchError> {
            todo!()
        }
    } else {
        // fallback impl
        pub fn cpu() -> Result<(), DispatchError> {
            todo!()
        }
    }
}

pub fn gpu() -> Result<(), DispatchError> {
    unimplemented!()
}
