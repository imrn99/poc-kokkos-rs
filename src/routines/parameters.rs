//! parallel statement parameterization code
//!
//! This module contains all code used to parameterize parallel statements. Some
//! parameters are direct Kokkos replicates while others are Rust-specific.
//!
//! Notable differences include:
//! - [ExecutionPolicy] struct: Instead of having multiple types of execution policies
//!   for each range, with re-occuring parameters, range specification is now a
//!   subparameter of execution policies.
//!

use std::ops::Range;

/// Execution Space enum. Used to specify the target device of execution for the
/// dispatch. Defaults to [ExecutionSpace::Serial]
#[derive(Default)]
pub enum ExecutionSpace {
    #[default]
    /// Default value. Execute the kernel sequentially.
    Serial,
    /// Target the CPU. Execute the kernel in parallel by using a feature-determined
    /// backend.
    DeviceCPU,
    /// Target the GPU. NOT SUPPORTED.
    DeviceGPU,
}

/// Range Policy enum. This holds information related to the looping structure
/// adopted by the routine.
pub enum RangePolicy<const N: usize> {
    /// 1D iteration range.
    RangePolicy(Range<usize>),
    /// N-dimensional iteration range.
    MDRangePolicy([Range<usize>; N]),
    /// Team-based iteration policy.
    TeamPolicy {
        /// Number of team.
        league_size: usize,
        /// Number of threads per team.
        team_size: usize,
        /// Number of vector
        vector_size: usize,
    },

    /// Policy used to ensure each team execute the body once and only once.
    PerTeam,
    /// Policy used to ensure each thread execute the body once and only once.
    PerThread,

    /// Medium-level depth. Can host further nests using vectors.
    TeamThreadRange,
    /// Medium-level depth. Can host further nests using vectors.
    TeamThreadMDRange,

    /// Medium-level depth. Cannot host further nests.
    TeamVectorRange,
    /// Medium-level depth. Cannot host further nests.
    TeamVectorMDRange,

    /// Inner-level depth. Cannot host further nests.
    ThreadVectorRange,
    /// Inner-level depth. Cannot host further nests.
    ThreadVectorMDRange,
}

/// Schedule enum. Used to set the workload scheduling policy.
/// Defaults to [Schedule::Static].
#[derive(Default)]
pub enum Schedule {
    #[default]
    /// Default value. Workload is divided once and split equally between
    /// computational ressources.
    Static,
    /// Dynamic scheduling. Workload is divided at start, split between computational
    /// ressources and work stealing is enabled.
    Dynamic,
}

/// Execution Policy enum. See Kokkos documentation for explanation on their model.
pub struct ExecutionPolicy<const N: usize> {
    /// Execution space targetted by the dispatch.
    pub space: ExecutionSpace,
    /// Iteration pattern used to handle the workload.
    pub range: RangePolicy<N>,
    /// Scheduling policy for the dispatch.
    pub schedule: Schedule,
}
