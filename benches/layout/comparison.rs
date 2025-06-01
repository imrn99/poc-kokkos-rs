use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{
    distr::{Distribution, Uniform},
    rngs::SmallRng,
    SeedableRng,
};

use poc_kokkos_rs::{
    functor::KernelArgs,
    routines::{
        parallel_for,
        parameters::{ExecutionPolicy, ExecutionSpace, RangePolicy, Schedule},
    },
    view::{parameters::Layout, ViewOwned},
};

// GEMM - worst case layout
fn f1(
    length: usize,
    aa_init: Vec<f64>,
    bb_init: Vec<f64>,
    cc_init: Vec<f64>,
    alpha: f64,
    beta: f64,
) {
    // worst case layout:
    // iterate on lines -> column-major layout (Left)
    // iterate on rows  -> line-major layout   (Right)
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Left, [length, length]);
    let mut bb = ViewOwned::new_from_data(bb_init, Layout::Right, [length, length]);
    let mut cc = ViewOwned::new_from_data(cc_init, Layout::Left, [length, length]);
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
    black_box(&cc);
}

// GEMM - usual case layout
fn f2(
    length: usize,
    aa_init: Vec<f64>,
    bb_init: Vec<f64>,
    cc_init: Vec<f64>,
    alpha: f64,
    beta: f64,
) {
    // best case layout:
    // iterate on lines -> line-major layout   (Right)
    // iterate on rows  -> column-major layout (Left)
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut bb = ViewOwned::new_from_data(bb_init, Layout::Right, [length, length]);
    let mut cc = ViewOwned::new_from_data(cc_init, Layout::Right, [length, length]);
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
    black_box(&cc);
}

// GEMM - best case layout
fn f3(
    length: usize,
    aa_init: Vec<f64>,
    bb_init: Vec<f64>,
    cc_init: Vec<f64>,
    alpha: f64,
    beta: f64,
) {
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut bb = ViewOwned::new_from_data(bb_init, Layout::Left, [length, length]);
    let mut cc = ViewOwned::new_from_data(cc_init, Layout::Right, [length, length]);
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
    black_box(&cc);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 10;
    let length = 2_usize.pow(DATA_SIZE);
    let seed: u64 = 9817498146784;
    let mut rng = SmallRng::seed_from_u64(seed);
    let range: Uniform<f64> = Uniform::new(0.0, 100.0).unwrap();
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

    let mut group = c.benchmark_group("gemm-layouts");
    group.bench_with_input(
        BenchmarkId::new("worst-layout", ""),
        &(
            length,
            aa_init.clone(),
            bb_init.clone(),
            cc_init.clone(),
            alpha,
            beta,
        ),
        |b, (length, aa_init, bb_init, cc_init, alpha, beta)| {
            b.iter(|| {
                f1(
                    *length,
                    aa_init.clone(),
                    bb_init.clone(),
                    cc_init.clone(),
                    *alpha,
                    *beta,
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("usual-layout", ""),
        &(
            length,
            aa_init.clone(),
            bb_init.clone(),
            cc_init.clone(),
            alpha,
            beta,
        ),
        |b, (length, aa_init, bb_init, cc_init, alpha, beta)| {
            b.iter(|| {
                f2(
                    *length,
                    aa_init.clone(),
                    bb_init.clone(),
                    cc_init.clone(),
                    *alpha,
                    *beta,
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("best-layout", ""),
        &(
            length,
            aa_init.clone(),
            bb_init.clone(),
            cc_init.clone(),
            alpha,
            beta,
        ),
        |b, (length, aa_init, bb_init, cc_init, alpha, beta)| {
            b.iter(|| {
                f3(
                    *length,
                    aa_init.clone(),
                    bb_init.clone(),
                    cc_init.clone(),
                    *alpha,
                    *beta,
                )
            })
        },
    );
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
