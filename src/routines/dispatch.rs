//! kernel dispatch code
//!
//! This module contains all code used to dispatch computational kernels
//! onto specified devices. Note that the documentation is feature-specific when the
//! items are, i.e. documentation is altered by enabled features.
//!
//! The methods desccribed in this module are not meant to be used directly, they are only
//! building blocks for the parallel statements.

#[cfg(any(doc, feature = "rayon", feature = "gpu"))]
use crate::functor::ForKernelType;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::{fmt::Display, ops::Range};

use super::parameters::{ExecutionPolicy, RangePolicy};
use crate::functor::{KernelArgs, SerialForKernelType};

// enums

/// Enum used to classify possible dispatch errors.
///
/// In all variants, the internal value is a description of the error.
#[derive(Debug)]
pub enum DispatchError {
    /// Error occured during serial dispatch.
    Serial(&'static str),
    /// Error occured during parallel CPU dispatch.
    CPU(&'static str),
    /// Error occured during GPU dispatch.
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
/// Technically, this should be replaced by a tiling function, for both serial and parallel
/// implementations.
fn recursive_loop<const N: usize>(ranges: &[Range<usize>; N], mut kernel: SerialForKernelType<N>) {
    // handles recursions
    fn inner<const N: usize>(
        current_depth: usize,
        ranges: &[Range<usize>; N],
        kernel: &mut SerialForKernelType<N>,
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

/// CPU dispatch routine of `for` statements. Does not depend on enabled feature(s).
///
/// The dispatch function execute the kernel accordingly to the directives contained in the
/// execution policy. The kernel signature does not vary according to enabled features as this
/// is the invariant fallback dispatch routine.
pub fn serial<const N: usize>(
    execp: ExecutionPolicy<N>,
    kernel: SerialForKernelType<N>,
) -> Result<(), DispatchError> {
    match execp.range {
        RangePolicy::RangePolicy(range) => {
            // serial, 1D range
            if N != 1 {
                return Err(DispatchError::Serial(
                    "Dispatch uses N>1 for a 1D RangePolicy",
                ));
            }
            range.into_iter().map(KernelArgs::Index1D).for_each(kernel)
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
    if #[cfg(feature = "threads")] {
        /// CPU dispatch routine of `for` statements. Implementation depends on enabled feature(s).
        ///
        /// The dispatch function execute the kernel accordingly to the directives contained in the
        /// execution policy. The kernel signature varies according to enabled features.
        ///
        /// ### Possible Kernel Signatures
        ///
        /// - `rayon` feature enabled: [`ForKernelType`]
        /// - `threads` feature enabled: `Box<impl Fn(KernelArgs<N>) + Send + Sync + 'a + Clone>`
        /// - no feature enabled: fall back to [`SerialForKernelType`]
        ///
        /// The `threads` implementation cannot currently use the generic [`ForKernelType`] because
        /// of the Clone requirement.
        ///
        /// **Current version**: `threads`
        pub fn cpu<'a, const N: usize>(
            execp: ExecutionPolicy<N>,
            kernel: Box<impl Fn(KernelArgs<N>) + Send + Sync + 'a + Clone>, // cannot be replaced by functor type bc of Clone
        ) -> Result<(), DispatchError> {
            match execp.range {
                RangePolicy::RangePolicy(range) => {
                    // serial, 1D range
                    if N != 1 {
                        return Err(DispatchError::Serial(
                            "Dispatch uses N>1 for a 1D RangePolicy",
                        ));
                    }
                    // compute chunk_size so that there is 1 chunk per thread
                    let chunk_size = range.len() / num_cpus::get() + 1;
                    let indices = range.collect::<Vec<usize>>();
                    // use scope to avoid 'static lifetime reqs
                    std::thread::scope(|s| {
                        let handles: Vec<_> = indices.chunks(chunk_size).map(|chunk| {
                            s.spawn(|| chunk.iter().map(|idx_ref| KernelArgs::Index1D(*idx_ref)).for_each(kernel.clone()))
                        }).collect();

                        for handle in handles {
                            handle.join().unwrap();
                        }
                    });
                }
                RangePolicy::MDRangePolicy(_) => {
                    // Kokkos does tiling to handle a MDRanges
                    unimplemented!()
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
                _ => todo!(),
            };
            Ok(())
        }
    } else if #[cfg(feature = "rayon")] {
        /// CPU dispatch routine of `for` statements. Implementation depends on enabled feature(s).
        ///
        /// The dispatch function execute the kernel accordingly to the directives contained in the
        /// execution policy. The kernel signature varies according to enabled features.
        ///
        /// ### Possible Kernel Signatures
        ///
        /// - `rayon` feature enabled: [`ForKernelType`]
        /// - `threads` feature enabled: `Box<impl Fn(KernelArgs<N>) + Send + Sync + 'a + Clone>`
        /// - no feature enabled: fall back to [`SerialForKernelType`]
        ///
        /// The `threads` implementation cannot currently use the generic [`ForKernelType`] because
        /// of the Clone requirement.
        ///
        /// **Current version**: `rayon`
        pub fn cpu<'a, const N: usize>(
            execp: ExecutionPolicy<N>,
            kernel: ForKernelType<N>,
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
                        .into_par_iter()
                        .map(KernelArgs::Index1D)
                        .for_each(kernel)
                }
                RangePolicy::MDRangePolicy(_) => {
                    // Kokkos does tiling to handle a MDRanges
                    unimplemented!()
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
                _ => todo!(),
            };
            Ok(())
        }
    } else {
        /// CPU dispatch routine of `for` statements. Implementation depends on enabled feature(s).
        ///
        /// The dispatch function execute the kernel accordingly to the directives contained in the
        /// execution policy. The kernel signature varies according to enabled features.
        ///
        /// ### Possible Kernel Signatures
        ///
        /// - `rayon` feature enabled: [`ForKernelType`]
        /// - `threads` feature enabled: `Box<impl Fn(KernelArgs<N>) + Send + Sync + 'a + Clone>`
        /// - no feature enabled: fall back to [`SerialForKernelType`]
        ///
        /// The `threads` implementation cannot currently use the generic [`ForKernelType`] because
        /// of the Clone requirement.
        ///
        /// **Current version**: no feature
        pub fn cpu<const N: usize>(
            execp: ExecutionPolicy<N>,
            kernel: SerialForKernelType<N>,
        ) -> Result<(), DispatchError> {
            serial(execp, kernel)
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "gpu")] {
        /// GPU Dispatch routine of `for` statements. UNIMPLEMENTED
        pub fn gpu<'a, const N: usize>(
            _execp: ExecutionPolicy<N>,
            _kernel: ForKernelType<N>,
        ) -> Result<(), DispatchError> {
            unimplemented!()
        }
    } else {
        /// GPU Dispatch routine of `for` statements. UNIMPLEMENTED
        pub fn gpu<const N: usize>(
            execp: ExecutionPolicy<N>,
            kernel: SerialForKernelType<N>,
        ) -> Result<(), DispatchError> {
            serial(execp, kernel)
        }
    }
}

// ~~~~~~
// Tests

mod tests {
    #[test]
    fn simple_range() {
        use super::*;
        use crate::{
            routines::parameters::{ExecutionSpace, Schedule},
            view::{parameters::Layout, ViewOwned},
        };
        // fixes warnings when testing using a parallel feature
        cfg_if::cfg_if! {
            if #[cfg(any(feature = "threads", feature = "rayon", feature = "gpu"))] {
                let mat = ViewOwned::new_from_data(vec![0.0; 15], Layout::Right, [15]);
            } else {
                let mut mat = ViewOwned::new_from_data(vec![0.0; 15], Layout::Right, [15]);
            }
        }
        let ref_mat = ViewOwned::new_from_data(vec![1.0; 15], Layout::Right, [15]);
        let rangep = RangePolicy::RangePolicy(0..15);
        let execp = ExecutionPolicy {
            space: ExecutionSpace::DeviceCPU,
            range: rangep,
            schedule: Schedule::default(),
        };

        // very messy way to write a kernel but it should work for now
        let kernel = Box::new(|arg: KernelArgs<1>| match arg {
            KernelArgs::Index1D(i) => mat.set([i], 1.0),
            KernelArgs::IndexND(_) => unimplemented!(),
            KernelArgs::Handle => unimplemented!(),
        });

        serial(execp, kernel).unwrap();

        assert_eq!(mat.raw_val().unwrap(), ref_mat.raw_val().unwrap());
    }

    #[test]
    fn simple_mdrange() {
        use super::*;
        use crate::{
            routines::parameters::{ExecutionSpace, Schedule},
            view::{parameters::Layout, ViewOwned},
        };
        // fixes warnings when testing using a parallel feature
        cfg_if::cfg_if! {
            if #[cfg(any(feature = "threads", feature = "rayon", feature = "gpu"))] {
                let mat = ViewOwned::new_from_data(vec![0.0; 150], Layout::Right, [10, 15]);
            } else {
                let mut mat = ViewOwned::new_from_data(vec![0.0; 150], Layout::Right, [10, 15]);
            }
        }
        let ref_mat = ViewOwned::new_from_data(vec![1.0; 150], Layout::Right, [10, 15]);
        let rangep = RangePolicy::MDRangePolicy([0..10, 0..15]);
        let execp = ExecutionPolicy {
            space: ExecutionSpace::DeviceCPU,
            range: rangep,
            schedule: Schedule::default(),
        };

        // very messy way to write a kernel but it should work for now
        let kernel = Box::new(|arg: KernelArgs<2>| match arg {
            KernelArgs::Index1D(_) => unimplemented!(),
            KernelArgs::IndexND([i, j]) => mat.set([i, j], 1.0),
            KernelArgs::Handle => unimplemented!(),
        });

        serial(execp, kernel).unwrap();

        assert_eq!(mat.raw_val().unwrap(), ref_mat.raw_val().unwrap());
    }

    #[test]
    fn dim1_mdrange() {
        use super::*;
        use crate::{
            routines::parameters::{ExecutionSpace, Schedule},
            view::{parameters::Layout, ViewOwned},
        };

        // fixes warnings when testing using a parallel feature
        cfg_if::cfg_if! {
            if #[cfg(any(feature = "threads", feature = "rayon", feature = "gpu"))] {
                let mat = ViewOwned::new_from_data(vec![0.0; 15], Layout::Right, [15]);
            } else {
                let mut mat = ViewOwned::new_from_data(vec![0.0; 15], Layout::Right, [15]);
            }
        }
        let ref_mat = ViewOwned::new_from_data(vec![1.0; 15], Layout::Right, [15]);
        #[allow(clippy::single_range_in_vec_init)]
        let rangep = RangePolicy::MDRangePolicy([0..15]);
        let execp = ExecutionPolicy {
            space: ExecutionSpace::DeviceCPU,
            range: rangep,
            schedule: Schedule::default(),
        };

        // very messy way to write a kernel but it should work for now
        let kernel = Box::new(|arg: KernelArgs<1>| match arg {
            KernelArgs::Index1D(_) => unimplemented!(),
            KernelArgs::IndexND(idx) => mat.set(idx, 1.0),
            KernelArgs::Handle => unimplemented!(),
        });

        serial(execp, kernel).unwrap();
        assert_eq!(mat.raw_val().unwrap(), ref_mat.raw_val().unwrap());
    }
}
