mod for_kernel;
mod policies;
mod reduce_kernel;
mod scan_kernel;

pub use for_kernel::{ForFunctor, parallel_for};
pub(crate) use policies::ExecutionPolicy;
pub use policies::{
    ExecutionSpace, MDRange, PerTeam, PerThread, Range, Schedule, TeamHandle, TeamPolicy,
    TeamThreadMDRange, TeamThreadRange, TeamVectorMDRange, TeamVectorRange, ThreadVectorMDRange,
    ThreadVectorRange,
};

// internal routines

// / Builds a N-depth nested loop executing a kernel using the N resulting indices.
// / Technically, this should be replaced by a tiling function, for both serial and parallel
// / implementations.
// fn recursive_loop<const N: usize>(ranges: &[usize; N], mut kernel: impl Fn([usize; N])) {
//     // handles recursions
//     fn inner<const N: usize>(
//         current_depth: usize,
//         ranges: &[usize; N],
//         kernel: &mut impl Fn([usize; N]),
//         indices: &mut [usize; N],
//     ) {
//         if current_depth == N {
//             // all loops unraveled
//             // call the kernel
//             kernel(*indices)
//         } else {
//             // loop on next dimension; update indices
//             // can we avoid a clone by passing a slice starting one element
//             // after the unraveled range ?
//             ranges[current_depth].clone().for_each(|i_current| {
//                 indices[current_depth] = i_current;
//                 inner(current_depth + 1, ranges, kernel, indices);
//             });
//         }
//     }

//     let mut indices = [0; N];
//     inner(0, ranges, &mut kernel, &mut indices);
// }
