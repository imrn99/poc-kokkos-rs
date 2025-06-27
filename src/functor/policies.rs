use std::marker::ConstParamTy;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, functor: F);

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, functor: F);

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, functor: F);
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

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, functor: F) {
        let _ = SCHEDULE;
        let Range(n) = self;

        (0..n).into_iter().for_each(|i| functor.execute(i));
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, functor: F) {
        let Range(n) = self;

        let n_threads = 4; // TODO: read it properly

        match SCHEDULE {
            #[cfg(not(feature = "rayon"))]
            Schedule::Static => {
                let chunk_size = n / n_threads + 1;

                std::thread::scope(|s| {
                    let mut handles = Vec::with_capacity(n_threads);
                    let f = &functor;
                    let blocks = (0..n_threads)
                        .map(|tid| tid * chunk_size..(tid * chunk_size + chunk_size).min(n));

                    for b in blocks {
                        let h = s.spawn(move || {
                            b.into_iter().for_each(|i| f.execute(i));
                        });
                        handles.push(h);
                    }

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            }
            #[cfg(not(feature = "rayon"))]
            Schedule::Dynamic => {
                unimplemented!("E: Dynamic dispatch isn't supported without rayon");
            }
            #[cfg(feature = "rayon")]
            Schedule::Static => {
                let chunk_size = n / n_threads + 1;

                // mimic static scheduling by generating one item to process per thread
                (0..n_threads)
                    .into_par_iter()
                    .map(|tid| tid * chunk_size..(tid * chunk_size + chunk_size).min(n))
                    .for_each(|b| {
                        b.into_iter().for_each(|i| functor.execute(i));
                    });
            }
            #[cfg(feature = "rayon")]
            Schedule::Dynamic => {
                (0..n).into_par_iter().for_each(|i| functor.execute(i));
            }
        }
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for MDRange<N> {
    type KernelArgType = [usize; N];

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}

impl ExecutionPolicy for TeamPolicy {
    type KernelArgType = TeamHandle;

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}
impl ExecutionPolicy for PerTeam {
    type KernelArgType = usize; // ?

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}
impl ExecutionPolicy for PerThread {
    type KernelArgType = usize; // ?

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}

impl ExecutionPolicy for TeamThreadRange {
    type KernelArgType = usize;

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for TeamThreadMDRange<N> {
    type KernelArgType = usize; // ?

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}

impl ExecutionPolicy for ThreadVectorRange {
    type KernelArgType = usize;

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for ThreadVectorMDRange<N> {
    type KernelArgType = usize; // ?

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}

impl ExecutionPolicy for TeamVectorRange {
    type KernelArgType = usize;

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }
}
impl<const N: usize> ExecutionPolicy for TeamVectorMDRange<N> {
    type KernelArgType = usize; // ?

    fn dispatch_seq<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_cpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
        todo!()
    }

    fn dispatch_gpu<const SCHEDULE: Schedule, F: ForFunctor<Self>>(self, _functor: F) {
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
