use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::view::{parameters::Layout, ViewOwned};
use rand::prelude::*;

// this bench is used to evaluate the cost of accessing views' data
// all benched functions contain 10^6 accesses.

// 1D vector access
fn f1(length: usize, indices: &[usize]) {
    let y: Vec<f64> = vec![0.0; length];
    let idx = &indices[0..length];

    for i in 0..1_000_000 {
        let tmp = y[idx[i % length]];
        black_box(tmp);
    }
}

// 1D view access
fn f1_b(length: usize, indices: &[usize]) {
    let v_y: ViewOwned<'_, 1, f64> =
        ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
    let idx = &indices[0..length];

    for i in 0..1_000_000 {
        let tmp = v_y[[idx[i % length]]];
        black_box(tmp);
    }
}

// 2D vector access
fn f2(length: usize, indices: &[usize]) {
    let y: Vec<f64> = vec![0.0; length * length];
    let idx = &indices[0..length];

    for i in 0..1_000 {
        for j in 1_000..2_000 {
            let tmp = unsafe { y.get_unchecked(idx[i % length] * length + idx[j % length]) };
            black_box(tmp);
        }
    }
}

// 2D view access
fn f2_b(length: usize, indices: &[usize]) {
    let v_y: ViewOwned<'_, 2, f64> =
        ViewOwned::new_from_data(vec![0.0; length * length], Layout::Right, [length, length]);
    let idx = &indices[0..length];

    for i in 0..1_000 {
        for j in 1_000..2_000 {
            let tmp = v_y[[idx[i % length], idx[j % length]]];
            black_box(tmp);
        }
    }
}

// 3D vector access
fn f3(length: usize, indices: &[usize]) {
    let y: Vec<f64> = vec![0.0; length * length * length];
    let idx = &indices[0..length];

    for i in 0..100 {
        for j in 100..200 {
            for k in 200..300 {
                // setting this as unchecked somehow produces the same perf as the 1D case
                // this does not work in 2D; perf stays the same
                // why?
                let tmp = unsafe {
                    y.get_unchecked(
                        idx[i % length] * length * length
                            + idx[j % length] * length
                            + idx[k % length],
                    )
                };
                black_box(tmp);
            }
        }
    }
}

// 3D view access
fn f3_b(length: usize, indices: &[usize]) {
    let v_y: ViewOwned<'_, 3, f64> = ViewOwned::new_from_data(
        vec![0.0; length * length * length],
        Layout::Right,
        [length, length, length],
    );
    let idx = &indices[0..length];

    for i in 0..100 {
        for j in 100..200 {
            for k in 200..300 {
                let tmp = v_y[[idx[i % length], idx[j % length], idx[k % length]]];
                black_box(tmp);
            }
        }
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 11; // 2048 length vector, 2048*2048 matrix
    let length = 2_usize.pow(DATA_SIZE);
    let mut rng = SmallRng::from_entropy();
    let indices: Vec<usize> = rand::seq::index::sample(&mut rng, length, length).into_vec();

    let mut group1 = c.benchmark_group("1D access");
    group1.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &(length, indices.clone()),
        |b, (n, i)| b.iter(|| f1(*n, i)),
    );
    group1.bench_with_input(
        BenchmarkId::new("View Access", ""),
        &(length, indices.clone()),
        |b, (n, i)| b.iter(|| f1_b(*n, i)),
    );
    group1.finish();

    let mut group2 = c.benchmark_group("2D access");
    group2.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &(length, indices.clone()),
        |b, (n, i)| b.iter(|| f2(*n, i)),
    );
    group2.bench_with_input(
        BenchmarkId::new("View Access", ""),
        &(length, indices.clone()),
        |b, (n, i)| b.iter(|| f2_b(*n, i)),
    );
    group2.finish();

    let mut group3 = c.benchmark_group("3D access");
    group3.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &(length, indices.clone()),
        |b, (n, i)| b.iter(|| f3(*n, i)),
    );
    group3.bench_with_input(
        BenchmarkId::new("View Access", ""),
        &(length, indices.clone()),
        |b, (n, i)| b.iter(|| f3_b(*n, i)),
    );
    group3.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
