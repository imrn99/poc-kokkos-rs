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

// Serial GEMV
fn f1(aa_init: Vec<f64>, x_init: Vec<f64>, y_init: Vec<f64>, alpha: f64, beta: f64) {
    let length = x_init.len();
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut x = ViewOwned::new_from_data(x_init, Layout::Right, [length]);
    let mut y = ViewOwned::new_from_data(y_init, Layout::Right, [length]);
    black_box(&mut aa);
    black_box(&mut x);
    black_box(&mut y);

    let execp = ExecutionPolicy {
        space: ExecutionSpace::Serial,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    // y = alpha * A * x + beta * y
    let gemv_kernel = |arg: KernelArgs<1>| match arg {
        KernelArgs::Index1D(i) => {
            let ax_i: f64 = (0..length).map(|j| aa.get([i, j]) * x.get([j])).sum();
            let val = alpha * ax_i + beta * y.get([i]);
            y.set([i], val);
        }
        KernelArgs::IndexND(_) => unimplemented!(),
        KernelArgs::Handle => unimplemented!(),
    };
    parallel_for(execp, gemv_kernel).unwrap();
    black_box(&y);
}

// DeviceCPU GEMV
fn f2(aa_init: Vec<f64>, x_init: Vec<f64>, y_init: Vec<f64>, alpha: f64, beta: f64) {
    let length = x_init.len();
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut x = ViewOwned::new_from_data(x_init, Layout::Right, [length]);
    let mut y = ViewOwned::new_from_data(y_init, Layout::Right, [length]);
    black_box(&mut aa);
    black_box(&mut x);
    black_box(&mut y);

    let execp = ExecutionPolicy {
        space: ExecutionSpace::DeviceCPU,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    // y = alpha * A * x + beta * y
    let gemv_kernel = |arg: KernelArgs<1>| match arg {
        KernelArgs::Index1D(i) => {
            let ax_i: f64 = (0..length).map(|j| aa.get([i, j]) * x.get([j])).sum();
            let val = alpha * ax_i + beta * y.get([i]);
            y.set([i], val);
        }
        KernelArgs::IndexND(_) => unimplemented!(),
        KernelArgs::Handle => unimplemented!(),
    };
    parallel_for(execp, gemv_kernel).unwrap();
    black_box(&y);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 12;
    let length = 2_usize.pow(DATA_SIZE);
    let seed: u64 = 9817498146784;
    let mut rng = SmallRng::seed_from_u64(seed);
    let range: Uniform<f64> = Uniform::new(0.0, 100.0).unwrap();
    let aa_init: Vec<f64> = (0..length * length)
        .map(|_| range.sample(&mut rng))
        .collect();
    let x_init: Vec<f64> = (0..length).map(|_| range.sample(&mut rng)).collect();
    let y_init: Vec<f64> = (0..length).map(|_| range.sample(&mut rng)).collect();
    let alpha: f64 = range.sample(&mut rng);
    let beta: f64 = range.sample(&mut rng);

    let mut group = c.benchmark_group("speedup-gemv");
    group.bench_with_input(
        BenchmarkId::new("exec-serial", ""),
        &(aa_init.clone(), x_init.clone(), y_init.clone(), alpha, beta),
        |b, (aa_init, x_init, y_init, alpha, beta)| {
            b.iter(|| {
                f1(
                    aa_init.clone(),
                    x_init.clone(),
                    y_init.clone(),
                    *alpha,
                    *beta,
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("exec-devicecpu", ""),
        &(aa_init.clone(), x_init.clone(), y_init.clone(), alpha, beta),
        |b, (aa_init, x_init, y_init, alpha, beta)| {
            b.iter(|| {
                f2(
                    aa_init.clone(),
                    x_init.clone(),
                    y_init.clone(),
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
