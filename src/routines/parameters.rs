//! parallel statement parameterization code
//!
//! This module contains all code used to parameterize parallel statements. Some
//! parameters are direct Kokkos replicates while others are Rust-specific.
//!
//! The most notable difference with Kokkos is the "flipped" structure of the
//!
//!

use std::ops::Range;

pub enum ExecutionSpace {
    Serial,
    Threads,
    Rayon,
    OpenMP,
}

/// Range Policy enum. This holds information related to the looping structure
/// adopted by the routine.
pub enum OuterRangePolicy<const N: usize> {
    /// 1D iteration range.
    RangePolicy(Range<usize>),
    /// N-dimensional iteration range.
    MDRangePolicy([Range<usize>; N]),
    /// Team-based iteration policy. TODO: Make it use const generics?
    TeamPolicy {
        /// Number of team.
        league_size: usize,
        /// Number of threads per team.
        team_size: usize,
        /// Number of vector
        vector_size: usize,
    },
}

/// Nested Policy enum. This is similar to [RangePolicy], but specific to nested parallel
/// patterns. TODO: Need to add inner data to each TYPE
pub enum InnerRangePolicy<const N: usize> {
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

pub trait RangePolicy {}

impl<const N: usize> RangePolicy for OuterRangePolicy<N> {}
impl<const N: usize> RangePolicy for InnerRangePolicy<N> {}

#[derive(Default)]
pub enum Schedule {
    #[default]
    Static,
    Dynamic,
}

pub struct ExecutionPolicy<R>
where
    R: RangePolicy,
{
    pub space: ExecutionSpace,
    pub range: R,
    pub schedule: Schedule,
}
