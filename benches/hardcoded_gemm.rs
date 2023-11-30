use atomic::Atomic;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::{
    distributions::{Distribution, Uniform},
    rngs::SmallRng,
    SeedableRng,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// hardcoded serial GEMM
fn serial_gemm(length: usize, aa: Vec<f64>, bb: Vec<f64>, cc: Vec<f64>, alpha: f64, beta: f64) {
    let mut aa = aa;
    let mut bb = bb;
    let mut cc = cc;
    black_box(&mut aa);
    black_box(&mut bb);
    black_box(&mut cc);

    for i in 0..length {
        for j in 0..length {
            // all b[k, j] for k values are adjacent in memory thanks to the LayoutLeft
            let ab_ij: f64 = (0..length)
                // unsafe access to keep things fair?
                .map(|k| unsafe {
                    aa.get_unchecked(i * length + k) * bb.get_unchecked(j * length + k)
                })
                .sum();
            let val: f64 = alpha * ab_ij + beta * cc[i * length + j];
            cc[i * length + j] = val;
        }
    }
    black_box(&cc);
}

// hardcoded rayon GEMM
fn gemm(
    length: usize,
    aa_init: Vec<f64>,
    bb_init: Vec<f64>,
    cc_init: Vec<f64>,
    alpha: f64,
    beta: f64,
) {
    let mut aa: Vec<Atomic<f64>> = aa_init
        .iter()
        .map(|val| atomic::Atomic::new(*val))
        .collect();
    let mut bb: Vec<Atomic<f64>> = bb_init
        .iter()
        .map(|val| atomic::Atomic::new(*val))
        .collect();
    let mut cc: Vec<Atomic<f64>> = cc_init
        .iter()
        .map(|val| atomic::Atomic::new(*val))
        .collect();
    black_box(&mut aa);
    black_box(&mut bb);
    black_box(&mut cc);

    // C = alpha * A * B + beta * C
    (0..length).into_par_iter().for_each(|i| {
        for j in 0..length {
            let ab_ij: f64 = (0..length)
                // unsafe access to keep things fair?
                .map(|k| unsafe {
                    aa.get_unchecked(i * length + k)
                        .load(atomic::Ordering::Relaxed)
                        * bb.get_unchecked(j * length + k)
                            .load(atomic::Ordering::Relaxed)
                })
                .sum();
            let val: f64 =
                alpha * ab_ij + beta * cc[i * length + j].load(atomic::Ordering::Relaxed);
            cc[i * length + j].store(val, atomic::Ordering::Relaxed);
        }
    });
    black_box(&cc);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
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

    let mut group = c.benchmark_group("gemm-hardcoded");
    group.bench_with_input(
        BenchmarkId::new("serial", ""),
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
                serial_gemm(
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
        BenchmarkId::new("rayon", ""),
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
                gemm(
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
