use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

use poc_kokkos_rs::{
    functor::{ExecutionSpace, Range, Schedule, parallel_for},
    view::{ViewOwned, parameters::Layout},
};

type FloatType = f64;

// GEMM - usual case layout
fn f1(
    length: usize,
    aa_init: Vec<FloatType>,
    bb_init: Vec<FloatType>,
    cc_init: Vec<FloatType>,
    alpha: FloatType,
    beta: FloatType,
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

    // C = alpha * A * B + beta * C
    parallel_for::<{ ExecutionSpace::DeviceCPU }, { Schedule::Static }, _, _>(
        None,
        Range(length),
        |i| {
            for j in 0..length {
                // all b[k, j] for k values are adjacent in memory thanks to the LayoutLeft
                let ab_ij: FloatType = (0..length).map(|k| aa.get([i, k]) * bb.get([k, j])).sum();
                let val: FloatType = alpha * ab_ij + beta * cc.get([i, j]);
                cc.set([i, j], val);
            }
        },
    );

    black_box(&cc);
}

// GEMM - best case layout
fn f2(
    length: usize,
    aa_init: Vec<FloatType>,
    bb_init: Vec<FloatType>,
    cc_init: Vec<FloatType>,
    alpha: FloatType,
    beta: FloatType,
) {
    let mut aa = ViewOwned::new_from_data(aa_init, Layout::Right, [length, length]);
    let mut bb = ViewOwned::new_from_data(bb_init, Layout::Left, [length, length]);
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
                let ab_ij: FloatType = (0..length).map(|k| aa.get([i, k]) * bb.get([k, j])).sum();
                let val: FloatType = alpha * ab_ij + beta * cc.get([i, j]);
                cc.set([i, j], val);
            }
        },
    );

    black_box(&cc);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm-sizes");
    for data_size in 5..11 {
        let length = 2_usize.pow(data_size);
        let seed: u64 = 9817498146784;
        let mut rng = SmallRng::seed_from_u64(seed);
        let range: Uniform<FloatType> = Uniform::new(0.0, 100.0).unwrap();
        let aa_init: Vec<FloatType> = (0..length * length)
            .map(|_| range.sample(&mut rng))
            .collect();
        let bb_init: Vec<FloatType> = (0..length * length)
            .map(|_| range.sample(&mut rng))
            .collect();
        let cc_init: Vec<FloatType> = (0..length * length)
            .map(|_| range.sample(&mut rng))
            .collect();
        let alpha: FloatType = range.sample(&mut rng);
        let beta: FloatType = range.sample(&mut rng);
        // f64 uses 8 bytes
        group.throughput(Throughput::Bytes((8 * length).pow(2) as u64));
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
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
