use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

use poc_kokkos_rs::{
    functor::{ExecutionSpace, Range, Schedule, parallel_for},
    view::{ViewOwned, parameters::Layout},
};

// Serial AXPY
fn f1(x_init: Vec<f64>, y_init: Vec<f64>, alpha: f64) {
    let length = x_init.len();
    let mut x = ViewOwned::new_from_data(x_init, Layout::Right, [length]);
    let mut y = ViewOwned::new_from_data(y_init, Layout::Right, [length]);
    black_box(&mut x);
    black_box(&mut y);

    // y = alpha * x + y
    parallel_for::<{ ExecutionSpace::DeviceCPU }, { Schedule::Static }, _, _>(
        None,
        Range(length),
        |i| {
            let val = alpha * x.get([i]) + y.get([i]);
            y.set([i], val);
        },
    );

    black_box(&y);
}

// DeviceCPU AXPY
fn f2(x_init: Vec<f64>, y_init: Vec<f64>, alpha: f64) {
    let length = x_init.len();
    let mut x = ViewOwned::new_from_data(x_init, Layout::Right, [length]);
    let mut y = ViewOwned::new_from_data(y_init, Layout::Right, [length]);
    black_box(&mut x);
    black_box(&mut y);

    // y = alpha * x + y
    parallel_for::<{ ExecutionSpace::DeviceCPU }, { Schedule::Static }, _, _>(
        None,
        Range(length),
        |i| {
            let val = alpha * x.get([i]) + y.get([i]);
            y.set([i], val);
        },
    );

    black_box(&y);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 20;
    let length = 2_usize.pow(DATA_SIZE);
    let seed: u64 = 9817498146784;
    let mut rng = SmallRng::seed_from_u64(seed);
    let range: Uniform<f64> = Uniform::new(0.0, 100.0).unwrap();
    let x_init: Vec<f64> = (0..length).map(|_| range.sample(&mut rng)).collect();
    let y_init: Vec<f64> = (0..length).map(|_| range.sample(&mut rng)).collect();
    let alpha: f64 = range.sample(&mut rng);

    let mut group = c.benchmark_group("speedup-axpy");
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
