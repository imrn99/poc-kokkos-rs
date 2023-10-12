use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::view::{parameters::Layout, ViewOwned};

// this bench is used to evaluate the cost of creating views

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

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    let data_size: u32 = 11; // 2048 length vector, 2048*2048 matrix

    let mut group1 = c.benchmark_group("1D view initialization");
    group1.bench_with_input(
        BenchmarkId::new("Standard Vector Allocation", ""),
        &data_size,
        |b, &n| b.iter(|| f1(n)),
    );
    group1.bench_with_input(
        BenchmarkId::new("Allocation & Init", ""),
        &data_size,
        |b, &n| b.iter(|| f1_b(n)),
    );
    group1.bench_with_input(
        BenchmarkId::new("Inline? Allocation & Init", ""),
        &data_size,
        |b, &n| b.iter(|| f1_bb(n)),
    );
    group1.bench_with_input(
        BenchmarkId::new("Default View Init", ""),
        &data_size,
        |b, &n| b.iter(|| f1_bbb(n)),
    );
    group1.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
