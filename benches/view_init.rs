use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::view::{parameters::Layout, ViewOwned};

// this bench is used to evaluate the cost of creating views

// 1D

// standard allocation
fn f1(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..1000 {
        let y: Vec<f64> = vec![0.0; length];
        black_box(y);
    }
}

// allocation & view init
fn f1_b(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..1000 {
        let y: Vec<f64> = vec![0.0; length];
        let v_y: ViewOwned<'_, 1, f64> = ViewOwned::new_from_data(y, Layout::Right, [length]);
        black_box(v_y);
    }
}

// inline? allocation & view init
fn f1_bb(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..1000 {
        let v_y: ViewOwned<'_, 1, f64> =
            ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
        black_box(v_y);
    }
}

// default view init
fn f1_bbb(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..1000 {
        let v_y: ViewOwned<'_, 1, f64> = ViewOwned::new(Layout::Right, [length]);
        black_box(v_y);
    }
}

// standard allocation
fn f2(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..100 {
        let y: Vec<f64> = vec![0.0; length * length];
        black_box(y);
    }
}

// 2D

// allocation & view init
fn f2_b(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..100 {
        let y: Vec<f64> = vec![0.0; length * length];
        let v_y: ViewOwned<'_, 2, f64> =
            ViewOwned::new_from_data(y, Layout::Right, [length, length]);
        black_box(v_y);
    }
}

// inline? allocation & view init
fn f2_bb(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..100 {
        let v_y: ViewOwned<'_, 2, f64> =
            ViewOwned::new_from_data(vec![0.0; length * length], Layout::Right, [length, length]);
        black_box(v_y);
    }
}

// default view init
fn f2_bbb(size: u32) {
    let length = 2_usize.pow(size);
    for _ in 0..100 {
        let v_y: ViewOwned<'_, 2, f64> = ViewOwned::new(Layout::Right, [length, length]);
        black_box(v_y);
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    let mut data_size: u32 = 11; // 2048 length vector, 2048*2048 matrix

    let mut group1 = c.benchmark_group("init-overhead-1D");
    group1.bench_with_input(BenchmarkId::new("vector-alloc", ""), &data_size, |b, &n| {
        b.iter(|| f1(n))
    });
    group1.bench_with_input(
        BenchmarkId::new("view-alloc-then-init", ""),
        &data_size,
        |b, &n| b.iter(|| f1_b(n)),
    );
    group1.bench_with_input(
        BenchmarkId::new("view-inline-alloc-init", ""),
        &data_size,
        |b, &n| b.iter(|| f1_bb(n)),
    );
    group1.bench_with_input(
        BenchmarkId::new("view-default-init", ""),
        &data_size,
        |b, &n| b.iter(|| f1_bbb(n)),
    );
    group1.finish();

    data_size = 10;

    let mut group2 = c.benchmark_group("init-overhead-2D");
    group2.bench_with_input(BenchmarkId::new("vector-alloc", ""), &data_size, |b, &n| {
        b.iter(|| f2(n))
    });
    group2.bench_with_input(
        BenchmarkId::new("view-alloc-then-init", ""),
        &data_size,
        |b, &n| b.iter(|| f2_b(n)),
    );
    group2.bench_with_input(
        BenchmarkId::new("view-inline-alloc-init", ""),
        &data_size,
        |b, &n| b.iter(|| f2_bb(n)),
    );
    group2.bench_with_input(
        BenchmarkId::new("view-default-init", ""),
        &data_size,
        |b, &n| b.iter(|| f2_bbb(n)),
    );
    group2.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
