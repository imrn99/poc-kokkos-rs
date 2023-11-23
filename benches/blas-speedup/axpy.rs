use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::{
    functor::KernelArgs,
    routines::{
        parallel_for,
        parameters::{ExecutionPolicy, ExecutionSpace, RangePolicy, Schedule},
    },
    view::{parameters::Layout, ViewOwned},
};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::SmallRng,
    SeedableRng,
};

// Serial AXPY
fn f1(x_init: Vec<f64>, y_init: Vec<f64>, alpha: f64) {
    let length = x_init.len();
    let mut x = ViewOwned::new_from_data(x_init, Layout::Right, [length]);
    let mut y = ViewOwned::new_from_data(y_init, Layout::Right, [length]);
    black_box(&mut x);
    black_box(&mut y);

    let execp = ExecutionPolicy {
        space: ExecutionSpace::Serial,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    // y = alpha * x + y
    let axpy_kernel = |arg: KernelArgs<1>| match arg {
        KernelArgs::Index1D(i) => {
            let val = alpha * x.get([i]) + y.get([i]);
            y.set([i], val);
        }
        KernelArgs::IndexND(_) => unimplemented!(),
        KernelArgs::Handle => unimplemented!(),
    };
    parallel_for(execp, axpy_kernel).unwrap();
    black_box(&y);
}

// DeviceCPU AXPY
fn f2(x_init: Vec<f64>, y_init: Vec<f64>, alpha: f64) {
    let length = x_init.len();
    let mut x = ViewOwned::new_from_data(x_init, Layout::Right, [length]);
    let mut y = ViewOwned::new_from_data(y_init, Layout::Right, [length]);
    black_box(&mut x);
    black_box(&mut y);

    let execp = ExecutionPolicy {
        space: ExecutionSpace::DeviceCPU,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    // y = alpha * x + y
    let axpy_kernel = |arg: KernelArgs<1>| match arg {
        KernelArgs::Index1D(i) => {
            let val = alpha * x.get([i]) + y.get([i]);
            y.set([i], val);
        }
        KernelArgs::IndexND(_) => unimplemented!(),
        KernelArgs::Handle => unimplemented!(),
    };

    parallel_for(execp, axpy_kernel).unwrap();
    black_box(&y);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 20;
    let length = 2_usize.pow(DATA_SIZE);
    let seed: u64 = 9817498146784;
    let mut rng = SmallRng::seed_from_u64(seed);
    let range: Uniform<f64> = rand::distributions::Uniform::new(0.0, 100.0);
    let x_init: Vec<f64> = (0..length).map(|_| range.sample(&mut rng)).collect();
    let y_init: Vec<f64> = (0..length).map(|_| range.sample(&mut rng)).collect();
    let alpha: f64 = range.sample(&mut rng);

    let mut group = c.benchmark_group("axpy");
    group.bench_with_input(
        BenchmarkId::new("exec-serial", ""),
        &(x_init.clone(), y_init.clone(), alpha),
        |b, (x_init, y_init, alpha)| b.iter(|| f1(x_init.clone(), y_init.clone(), *alpha)),
    );
    group.bench_with_input(
        BenchmarkId::new("exec-devicecpu", ""),
        &(x_init.clone(), y_init.clone(), alpha),
        |b, (x_init, y_init, alpha)| b.iter(|| f2(x_init.clone(), y_init.clone(), *alpha)),
    );
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
