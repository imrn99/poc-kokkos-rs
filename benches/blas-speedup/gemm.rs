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

// Serial GEMM
fn f1(
    length: usize,
    aa_init: Vec<f64>,
    bb_init: Vec<f64>,
    cc_init: Vec<f64>,
    alpha: f64,
    beta: f64,
) {
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut bb = ViewOwned::new_from_data(bb_init, Layout::Left, [length, length]); // optimal layout since we iterate inside columns :)
    let mut cc = ViewOwned::new_from_data(cc_init, Layout::Right, [length, length]);
    black_box(&mut aa);
    black_box(&mut bb);
    black_box(&mut cc);

    // C = alpha * A * B + beta * C
    parallel_for::<{ ExecutionSpace::DeviceCPU }, { Schedule::Static }, _, _>(
        None,
        Range(length),
        |i| {
            for j in 0..length {
                // all b[k, j] for k values are adjacent in memory thanks to the LayoutLeft
                let ab_ij: f64 = (0..length).map(|k| aa.get([i, k]) * bb.get([k, j])).sum();
                let val: f64 = alpha * ab_ij + beta * cc.get([i, j]);
                cc.set([i, j], val);
            }
        },
    );

    black_box(&cc);
}

// DeviceCPU GEMM
fn f2(
    length: usize,
    aa_init: Vec<f64>,
    bb_init: Vec<f64>,
    cc_init: Vec<f64>,
    alpha: f64,
    beta: f64,
) {
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut bb = ViewOwned::new_from_data(bb_init, Layout::Left, [length, length]); // optimal layout since we iterate inside columns :)
    let mut cc = ViewOwned::new_from_data(cc_init, Layout::Right, [length, length]);
    black_box(&mut aa);
    black_box(&mut bb);
    black_box(&mut cc);

    // C = alpha * A * B + beta * C
    parallel_for::<{ ExecutionSpace::DeviceCPU }, { Schedule::Static }, _, _>(
        None,
        Range(length),
        |i| {
            for j in 0..length {
                // all b[k, j] for k values are adjacent in memory thanks to the LayoutLeft
                let ab_ij: f64 = (0..length).map(|k| aa.get([i, k]) * bb.get([k, j])).sum();
                let val: f64 = alpha * ab_ij + beta * cc.get([i, j]);
                cc.set([i, j], val);
            }
        },
    );

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

    let mut group = c.benchmark_group("speedup-gemm");
    group.bench_with_input(
        BenchmarkId::new("exec-serial", ""),
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
        BenchmarkId::new("exec-devicecpu", ""),
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
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
