//!
//!
//!
//!

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::{fmt::Display, ops::Range};

use super::parameters::{ExecutionPolicy, RangePolicy};
use crate::functor::{ForKernel, KernelArgs};

// enums

#[derive(Debug)]
pub enum DispatchError {
    Serial(&'static str),
    CPU(&'static str),
    GPU(&'static str),
}

impl Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::Serial(desc) => write!(f, "error during serial dispatch: {desc}"),
            DispatchError::CPU(desc) => write!(f, "error during cpu dispatch: {desc}"),
            DispatchError::GPU(desc) => write!(f, "error during gpu dispatch: {desc}"),
        }
    }
}

impl std::error::Error for DispatchError {
    // may be useful in case of an error coming from an std call
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

// dispatch routines

// internal routines

/// Builds a N-depth nested loop executing a kernel using the N resulting indices.
fn recursive_loop<const N: usize>(ranges: &[Range<usize>; N], mut kernel: ForKernel<N>) {
    // handles recursions
    fn inner<const N: usize>(
        current_depth: usize,
        ranges: &[Range<usize>; N],
        kernel: &mut ForKernel<N>,
        indices: &mut [usize; N],
    ) {
        if current_depth == N {
            // all loops unraveled
            // call the kernel
            kernel(KernelArgs::IndexND(*indices))
        } else {
            // loop on next dimension; update indices
            // can we avoid a clone by passing a slice starting one element
            // after the unraveled range ?
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

/// Dispatch routine for serial backend. Depending on the future parallel dispatch implem, this
/// could be deleted.
pub fn serial<const N: usize>(
    execp: ExecutionPolicy<N>,
    kernel: ForKernel<N>,
) -> Result<(), DispatchError> {
    match execp.range {
        RangePolicy::RangePolicy(range) => {
            // serial, 1D range
            if N != 1 {
                return Err(DispatchError::Serial(
                    "Dispatch uses N>1 for a 1D RangePolicy",
                ));
            }
            // making indices N-sized arrays is necessary, even with the assertion...
            range
                .into_iter()
                .map(|i| KernelArgs::Index1D(i))
                .for_each(kernel)
        }
        RangePolicy::MDRangePolicy(ranges) => {
            // Kokkos does tiling to handle a MDRanges, in the case of serial
            // execution, we simply do the nested loop
            recursive_loop(&ranges, kernel) // macros would pbly be more efficient
        }
        RangePolicy::TeamPolicy {
            league_size: _, // number of teams; akin to # of work items/batches
            team_size: _,   // number of threads per team; ignored in serial dispatch
            vector_size: _, // possible third dim parallelism; ignored in serial dispatch?
        } => {
            // interpret # of teams as # of work items (chunks);
            // necessary because serial dispatch is the fallback implementation
            // we ignore team size & vector size? since there's no parallelism here

            // is it even possible to use chunks? It would require either:
            //  - awareness of used external data
            //  - owning the used data; maybe in the TeamPolicy struct
            // 2nd option is the more plausible but it creates issues when accessing
            // multiple views for example; It also seems incompatible with the paradigm

            // -> build a team handle & let the user write its kernel using it
            todo!()
        }
        RangePolicy::PerTeam => {
            // used inside a team dispatch
            // executes the kernel once per team
            todo!()
        }
        RangePolicy::PerThread => {
            // used inside a team dispatch
            // executes the kernel once per threads of the team
            todo!()
        }
        RangePolicy::TeamThreadRange => {
            // same as RangePolicy but inside a team
            todo!()
        }
        RangePolicy::TeamThreadMDRange => {
            // same as MDRangePolicy but inside a team
            todo!()
        }
        RangePolicy::TeamVectorRange => todo!(),
        RangePolicy::TeamVectorMDRange => todo!(),
        RangePolicy::ThreadVectorRange => todo!(),
        RangePolicy::ThreadVectorMDRange => todo!(),
    };
    Ok(())
}

#[cfg(feature = "threads")]
/// Dispatch routine for CPU parallelization using [std::thread].
pub fn cpu<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
where
    F: FnMut([usize; N]) + Sync + Send,
{
    todo!()
}

#[cfg(feature = "rayon")]
/// Dispatch routine for CPU parallelization using [rayon][https://docs.rs/rayon/latest/rayon/].
pub fn cpu<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
where
    F: Fn([usize; N]) + Sync + Send,
{
    match execp.range {
        RangePolicy::RangePolicy(range) => {
            // serial, 1D range
            if N != 1 {
                return Err(DispatchError::Serial(
                    "Dispatch uses N>1 for a 1D RangePolicy",
                ));
            }
            // making indices N-sized arrays is necessary, even with the assertion...
            range.into_par_iter().map(|i| [i; N]).for_each(kernel)
        }
        RangePolicy::MDRangePolicy(ranges) => {
            // Kokkos does tiling to handle a MDRanges,
            todo!()
        }
        RangePolicy::TeamPolicy {
            league_size: _, // number of teams; akin to # of work items/batches
            team_size: _,   // number of threads per team; ignored in serial dispatch
            vector_size: _, // possible third dim parallelism; ignored in serial dispatch?
        } => {
            // interpret # of teams as # of work items (chunks);
            // necessary because serial dispatch is the fallback implementation
            // we ignore team size & vector size? since there's no parallelism here

            // is it even possible to use chunks? It would require either:
            //  - awareness of used external data
            //  - owning the used data; maybe in the TeamPolicy struct
            // 2nd option is the more plausible but it creates issues when accessing
            // multiple views for example; It also seems incompatible with the paradigm

            // -> build a team handle & let the user write its kernel using it
            todo!()
        }
        RangePolicy::PerTeam => {
            // used inside a team dispatch
            // executes the kernel once per team
            todo!()
        }
        RangePolicy::PerThread => {
            // used inside a team dispatch
            // executes the kernel once per threads of the team
            todo!()
        }
        RangePolicy::TeamThreadRange => {
            // same as RangePolicy but inside a team
            todo!()
        }
        RangePolicy::TeamThreadMDRange => {
            // same as MDRangePolicy but inside a team
            todo!()
        }
        RangePolicy::TeamVectorRange => todo!(),
        RangePolicy::TeamVectorMDRange => todo!(),
        RangePolicy::ThreadVectorRange => todo!(),
        RangePolicy::ThreadVectorMDRange => todo!(),
    };
    Ok(())
}

#[cfg(not(any(feature = "rayon", feature = "threads")))]
/// Fallback dispatch routine for CPU execution. Calls the serial dispatch routine.
pub fn cpu<const N: usize>(
    execp: ExecutionPolicy<N>,
    kernel: ForKernel<N>,
) -> Result<(), DispatchError> {
    serial(execp, kernel)
}

pub fn gpu<const N: usize>(
    _execp: ExecutionPolicy<N>,
    _kernel: ForKernel<N>,
) -> Result<(), DispatchError> {
    unimplemented!()
}

mod tests {
    use super::*;

    use crate::{
        routines::parameters::{ExecutionSpace, Schedule},
        view::{parameters::Layout, ViewOwned},
    };

    #[test]
    fn simple_range() {
        let mut mat = ViewOwned::new_from_data(vec![0.0; 15], Layout::Right, [15]);
        let ref_mat = ViewOwned::new_from_data(vec![1.0; 15], Layout::Right, [15]);
        let rangep = RangePolicy::RangePolicy(0..15);
        let execp = ExecutionPolicy {
            space: ExecutionSpace::default(),
            range: rangep,
            schedule: Schedule::default(),
        };

        // very messy way to write a kernel but it should work for now
        let kernel = Box::new(|arg: KernelArgs<1>| match arg {
            KernelArgs::Index1D(i) => mat[[i]] = 1.0,
            KernelArgs::IndexND(_) => unimplemented!(),
            KernelArgs::Handle => unimplemented!(),
        });

        serial(execp, kernel).unwrap();

        assert_eq!(mat, ref_mat);
    }

    #[test]
    fn simple_mdrange() {
        let mut mat = ViewOwned::new_from_data(vec![0.0; 150], Layout::Right, [10, 15]);
        let ref_mat = ViewOwned::new_from_data(vec![1.0; 150], Layout::Right, [10, 15]);
        let rangep = RangePolicy::MDRangePolicy([0..10, 0..15]);
        let execp = ExecutionPolicy {
            space: ExecutionSpace::default(),
            range: rangep,
            schedule: Schedule::default(),
        };

        // very messy way to write a kernel but it should work for now
        let kernel = Box::new(|arg: KernelArgs<2>| match arg {
            KernelArgs::Index1D(_) => unimplemented!(),
            KernelArgs::IndexND([i, j]) => mat[[i, j]] = 1.0,
            KernelArgs::Handle => unimplemented!(),
        });

        serial(execp, kernel).unwrap();

        assert_eq!(mat, ref_mat);
    }

    #[test]
    fn dim1_mdrange() {
        let mut mat = ViewOwned::new_from_data(vec![0.0; 15], Layout::Right, [15]);
        let ref_mat = ViewOwned::new_from_data(vec![1.0; 15], Layout::Right, [15]);
        #[allow(clippy::single_range_in_vec_init)]
        let rangep = RangePolicy::MDRangePolicy([0..15]);
        let execp = ExecutionPolicy {
            space: ExecutionSpace::default(),
            range: rangep,
            schedule: Schedule::default(),
        };

        // very messy way to write a kernel but it should work for now
        let kernel = Box::new(|arg: KernelArgs<1>| match arg {
            KernelArgs::Index1D(_) => unimplemented!(),
            KernelArgs::IndexND(idx) => mat[idx] = 1.0,
            KernelArgs::Handle => unimplemented!(),
        });

        serial(execp, kernel).unwrap();

        assert_eq!(mat, ref_mat);
    }
}
