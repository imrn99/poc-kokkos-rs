use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::view::{parameters::Layout, ViewOwned};
use rand::prelude::*;

// this bench is used to evaluate the cost of accessing views' data
// all benched functions contain 10^3 accesses.

// 1D vector access
fn f1(length: usize, indices: &[usize]) {
    let y: Vec<f64> = vec![0.0; length];
    let idx = &indices[0..length];

    idx.iter().for_each(|i| {
        let tmp = y[*i];
        black_box(tmp);
    })
}

// 1D view access
fn f1_b(length: usize, indices: &[usize]) {
    let v_y: ViewOwned<'_, 1, f64> =
        ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
    let idx = &indices[0..length];

    idx.iter().for_each(|i| {
        let tmp = v_y[[*i]];
        black_box(tmp);
    })
}

// 2D vector access
fn f2(length: usize, indices: &[(usize, usize)]) {
    let y: Vec<f64> = vec![0.0; length * length];
    let idx = &indices[0..length];

    idx.iter().for_each(|(i, j)| {
        let tmp = unsafe { y.get_unchecked(i * length + j) };
        black_box(tmp);
    });
}

// 2D view access
fn f2_b(length: usize, indices: &[(usize, usize)]) {
    let v_y: ViewOwned<'_, 2, f64> =
        ViewOwned::new_from_data(vec![0.0; length * length], Layout::Right, [length, length]);
    let idx = &indices[0..length];

    idx.iter().for_each(|(i, j)| {
        let tmp = v_y[[*i, *j]];
        black_box(tmp);
    })
}

// 3D vector access
fn f3(length: usize, indices: &[(usize, usize, usize)]) {
    let y: Vec<f64> = vec![0.0; length * length * length];
    let idx = &indices[0..length];

    idx.iter().for_each(|(i, j, k)| {
        // WARNING
        // For some reason, if the access is not dereferenced, it gets optimized away
        // You can verify it by running the benchmark twice:
        // - once with the blackbox, without the deref operator *
        // - once without the blackbox, with the deref operator *
        // both yields the same result;
        // the blackbox is supposed to prevent this, works in the 2D case, but not here
        let tmp = *unsafe { y.get_unchecked(i * length * length + j * length + k) };
        black_box(tmp);
    })
}

// 3D view access
fn f3_b(length: usize, indices: &[(usize, usize, usize)]) {
    let v_y: ViewOwned<'_, 3, f64> = ViewOwned::new_from_data(
        vec![0.0; length * length * length],
        Layout::Right,
        [length, length, length],
    );
    let idx = &indices[0..length];

    idx.iter().for_each(|(i, j, k)| {
        let tmp = v_y[[*i, *j, *k]];
        black_box(tmp);
    })
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 11; // 2048 length vector, 2048*2048 matrix
    let length = 2_usize.pow(DATA_SIZE);
    let mut rng = SmallRng::from_entropy();
    let indices1: Vec<usize> = rand::seq::index::sample(&mut rng, length, length).into_vec();
    let indices1b: Vec<usize> = rand::seq::index::sample(&mut rng, length, length).into_vec();
    let indices1bb: Vec<usize> = rand::seq::index::sample(&mut rng, length, length).into_vec();

    let indices2: Vec<(usize, usize)> = indices1
        .iter()
        .zip(indices1b.iter())
        .map(|(i1, i2)| (*i1, *i2))
        .collect();

    let indices3: Vec<(usize, usize, usize)> = indices1
        .iter()
        .zip(indices1b.iter())
        .zip(indices1bb.iter())
        .map(|((i1, i2), i3)| (*i1, *i2, *i3))
        .collect();

    let mut group1 = c.benchmark_group("1D access");
    group1.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &(length, indices1.clone()),
        |b, (n, i)| b.iter(|| f1(*n, i)),
    );
    group1.bench_with_input(
        BenchmarkId::new("View Access", ""),
        &(length, indices1),
        |b, (n, i)| b.iter(|| f1_b(*n, i)),
    );
    group1.finish();

    let mut group2 = c.benchmark_group("2D access");
    group2.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &(length, (indices2.clone())),
        |b, (n, i)| b.iter(|| f2(*n, i)),
    );
    group2.bench_with_input(
        BenchmarkId::new("View Access", ""),
        &(length, (indices2)),
        |b, (n, i)| b.iter(|| f2_b(*n, i)),
    );
    group2.finish();

    let mut group3 = c.benchmark_group("3D access");
    group3.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &(length, indices3.clone()),
        |b, (n, i)| b.iter(|| f3(*n, i)),
    );
    group3.bench_with_input(
        BenchmarkId::new("View Access", ""),
        &(length, indices3),
        |b, (n, i)| b.iter(|| f3_b(*n, i)),
    );
    group3.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
