use std::marker::ConstParamTy;

use crate::functor::ForFunctor;

// -- scheduling

/// Scheduling enum. CURRENTLY IGNORED.
///
/// Used to set the workload scheduling policy. Defaults to [Schedule::Static].
#[derive(ConstParamTy, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    #[default]
    /// Default value. Workload is divided once and split equally between
    /// computational ressources.
    Static,
    /// Dynamic scheduling. Workload is divided at start, split between computational
    /// ressources and work stealing is enabled.
    Dynamic,
}

// -- exec space

/// Execution Space enum.
///
/// Used to specify the target device of execution for the dispatch.
/// Defaults to [ExecutionSpace::Serial].
#[derive(ConstParamTy, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionSpace {
    #[default]
    /// Default value. Execute the kernel sequentially.
    Serial,
    /// Target the CPU. Execute the kernel in parallel by using a feature-determined
    /// backend.
    DeviceCPU,
    /// Target the GPU. UNIMPLEMENTED.
    DeviceGPU,
}

// -- exec policy

pub(crate) trait ExecutionPolicy
where
    Self: Sized,
{
    type KernelArgType;

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    );
}

pub struct TeamHandle {
    league_rank: usize,
    team_rank: usize,
    team_size: usize,
}

pub struct Range(pub usize);
pub struct MDRange<const N: usize>(pub [usize; N]);

pub struct TeamPolicy(pub usize, pub usize, pub usize);
pub struct PerTeam;
pub struct PerThread;

pub struct TeamThreadRange(pub TeamHandle, pub usize);
pub struct TeamThreadMDRange<const N: usize>(pub TeamHandle, pub [usize; N]);

pub struct TeamVectorRange(pub TeamHandle, pub usize);
pub struct TeamVectorMDRange<const N: usize>(pub TeamHandle, pub [usize; N]);

pub struct ThreadVectorRange(pub TeamHandle, pub usize);
pub struct ThreadVectorMDRange<const N: usize>(pub TeamHandle, pub [usize; N]);

impl ExecutionPolicy for Range {
    type KernelArgType = usize;

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for MDRange<N> {
    type KernelArgType = [usize; N];

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}

impl ExecutionPolicy for TeamPolicy {
    type KernelArgType = TeamHandle;

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}
impl ExecutionPolicy for PerTeam {
    type KernelArgType = usize; // ?

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}
impl ExecutionPolicy for PerThread {
    type KernelArgType = usize; // ?

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}

impl ExecutionPolicy for TeamThreadRange {
    type KernelArgType = usize;

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for TeamThreadMDRange<N> {
    type KernelArgType = usize; // ?

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}

impl ExecutionPolicy for ThreadVectorRange {
    type KernelArgType = usize;

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for ThreadVectorMDRange<N> {
    type KernelArgType = usize; // ?

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}

impl ExecutionPolicy for TeamVectorRange {
    type KernelArgType = usize;

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for TeamVectorMDRange<N> {
    type KernelArgType = usize; // ?

    fn dispatch<
        const EXECUTION_SPACE: ExecutionSpace,
        const SCHEDULE: Schedule,
        F: ForFunctor<Self>,
    >(
        functor: F,
    ) {
        todo!()
    }
}

impl From<usize> for Range {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl<const N: usize> From<[usize; N]> for MDRange<N> {
    fn from(value: [usize; N]) -> Self {
        Self(value)
    }
}
