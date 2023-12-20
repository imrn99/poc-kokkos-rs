//use poc_kokkos_rs::ffi;

use std::hint::black_box;

use poc_kokkos_rs::{
    functor::KernelArgs,
    routines::{
        parallel_for,
        parameters::{ExecutionPolicy, ExecutionSpace, RangePolicy, Schedule},
    },
    view::{parameters::Layout, View},
};
use rand::{distributions::Uniform, prelude::*, rngs::SmallRng, SeedableRng};

fn main() {
    // ffi::say_hello();
    // println!("Hello from Rust!");
    // ffi::say_many_hello()

    // inits
    const DATA_SIZE: u32 = 10;
    let length = 2_usize.pow(DATA_SIZE);
    let seed: u64 = 9817498146784;
    let mut rng = SmallRng::seed_from_u64(seed);
    let range: Uniform<f64> = rand::distributions::Uniform::new(0.0, 100.0);
    let aa_init: Vec<f64> = (0..length * length)
        .map(|_| range.sample(&mut rng))
        .collect();
    let bb_init: Vec<f64> = (0..length * length)
        .map(|_| range.sample(&mut rng))
        .collect();
    let cc_init: Vec<f64> = (0..length * length)
        .map(|_| range.sample(&mut rng))
        .collect();
    let alpha: f64 = range.sample(&mut rng);
    let beta: f64 = range.sample(&mut rng);

    // inits again
    let mut aa = View::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut bb = View::new_from_data(bb_init, Layout::Left, [length, length]); // optimal layout since we iterate inside columns :)
    let mut cc = View::new_from_data(cc_init, Layout::Right, [length, length]);
    black_box(&mut aa);
    black_box(&mut bb);
    black_box(&mut cc);

    let execp = ExecutionPolicy {
        space: ExecutionSpace::DeviceCPU,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    // C = alpha * A * B + beta * C
    let gemm_kernel = |arg: KernelArgs<1>| match arg {
        // lines
        KernelArgs::Index1D(i) => {
            // cols
            for j in 0..length {
                // all b[k, j] for k values are adjacent in memory thanks to the LayoutLeft
                let ab_ij: f64 = (0..length).map(|k| aa.get([i, k]) * bb.get([k, j])).sum();
                let val: f64 = alpha * ab_ij + beta * cc.get([i, j]);
                cc.set([i, j], val);
            }
        }
        KernelArgs::IndexND(_) => unimplemented!(),
        KernelArgs::Handle => unimplemented!(),
    };
    parallel_for(execp, gemm_kernel).unwrap();
}
