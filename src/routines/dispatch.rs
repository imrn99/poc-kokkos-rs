//!
//!
//!
//!

use std::{fmt::Display, ops::Range};

use super::parameters::{ExecutionPolicy, RangePolicy};

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
        match self {
            DispatchError::Serial(_) => None,
            DispatchError::CPU(_) => None,
            DispatchError::GPU(_) => None,
        }
    }
}

// dispatch routines

// internal routines

/// Builds a N-depth nested loop executing a kernel using the N resulting indices.
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

/// Dispatch routine for serial backend.
pub fn serial<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
where
    // [usize; N] should eventually be replaced by a KernelArgs trait that
    // supports the different kind of signatures you find in Kokkos
    F: FnMut([usize; N]), // FnMut is the closure trait taken by for_each method
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
            range.into_iter().map(|i| [i; N]).for_each(kernel)
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

cfg_if::cfg_if! {
    if #[cfg(threads)] {
        // OS threads backend
        pub fn cpu<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
        where
            F: FnMut([usize; N]), {
            todo!()
        }
    } else if #[cfg(rayon)] {
        // rayon backend
        pub fn cpu<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
        where
            F: FnMut([usize; N]), {
            todo!()
        }
    } else {
        // fallback impl: serial
        pub fn cpu<const N: usize, F>(execp: ExecutionPolicy<N>, kernel: F) -> Result<(), DispatchError>
        where
            F: FnMut([usize; N]), {
            serial(execp, kernel)
        }
    }
}

pub fn gpu<const N: usize, F>(_execp: ExecutionPolicy<N>, _kernel: F) -> Result<(), DispatchError>
where
    F: FnMut([usize; N]),
{
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use crate::{
        routines::parameters::{ExecutionSpace, Schedule},
        view::{parameters::Layout, ViewOwned},
    };

    use super::*;

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

        serial(execp, |[i]| {
            mat[[i]] = 1.0;
        })
        .unwrap();

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

        serial(execp, |[i, j]| {
            mat[[i, j]] = 1.0;
        })
        .unwrap();

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

        serial(execp, |[i]| {
            mat[[i]] = 1.0;
        })
        .unwrap();

        assert_eq!(mat, ref_mat);
    }
}
