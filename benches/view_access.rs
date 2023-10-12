use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::view::{parameters::Layout, ViewOwned};

// this bench is used to evaluate the cost of accessing views' data

// 1D vector access
fn f1(size: u32) {
    let length = 2_usize.pow(size);
    let mut y: Vec<f64> = vec![0.0; length];
    black_box(&mut y); // necessary?

    for i in 0..1_000_000 {
        // smth like that to prevent bias from cache usage?
        black_box(y[(i << 0x01) % length]);
    }
}

// 1D view access
fn f1_b(size: u32) {
    let length = 2_usize.pow(size);
    let mut v_y: ViewOwned<'_, 1, f64> =
        ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
    black_box(&mut v_y); // necessary?

    for i in 0..1_000_000 {
        black_box(v_y[[(i << 0x01) % length]]);
    }
}

// 2D vector access
fn f2(size: u32) {
    let length = 2_usize.pow(size);
    let mut y: Vec<f64> = vec![0.0; length * length];
    black_box(&mut y); // necessary?

    // simulate 2 indexes ?
    for i in 0..1_000 {
        for j in 0..1_000 {
            black_box(y[((i << 0x01) % length) * length + ((j >> 0x01) % length)]);
        }
    }
}

// 2D view access
fn f2_b(size: u32) {
    let length = 2_usize.pow(size);
    let mut v_y: ViewOwned<'_, 2, f64> =
        ViewOwned::new_from_data(vec![0.0; length * length], Layout::Right, [length, length]);
    black_box(&mut v_y); // necessary?

    for i in 0..1_000 {
        for j in 0..1_000 {
            black_box(v_y[[(i << 0x01) % length, ((j >> 0x01) % length)]]);
        }
    }
}

// 3D vector access
fn f3(size: u32) {
    let length = 2_usize.pow(size);
    let mut y: Vec<f64> = vec![0.0; length * length * length];
    black_box(&mut y); // necessary?

    // simulate 2 indexes ?
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                black_box(
                    y[((i << 0x01) % length) * length * length
                        + ((j >> 0x01) % length) * length
                        + ((k << 0x01) % length)],
                );
            }
        }
    }
}

// 3D view access
fn f3_b(size: u32) {
    let length = 2_usize.pow(size);
    let mut v_y: ViewOwned<'_, 3, f64> = ViewOwned::new_from_data(
        vec![0.0; length * length * length],
        Layout::Right,
        [length, length, length],
    );
    black_box(&mut v_y); // necessary?

    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                black_box(
                    v_y[[
                        (i << 0x01) % length,
                        (j >> 0x01) % length,
                        (k << 0x01) % length,
                    ]],
                );
            }
        }
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    let data_size: u32 = 11; // 2048 length vector, 2048*2048 matrix

    let mut group1 = c.benchmark_group("1D access");
    group1.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &data_size,
        |b, &n| b.iter(|| f1(n)),
    );
    group1.bench_with_input(BenchmarkId::new("View Access", ""), &data_size, |b, &n| {
        b.iter(|| f1_b(n))
    });
    group1.finish();

    let mut group2 = c.benchmark_group("2D access");
    group2.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &data_size,
        |b, &n| b.iter(|| f2(n)),
    );
    group2.bench_with_input(BenchmarkId::new("View Access", ""), &data_size, |b, &n| {
        b.iter(|| f2_b(n))
    });
    group2.finish();

    let mut group3 = c.benchmark_group("3D access");
    group3.bench_with_input(
        BenchmarkId::new("Vector Access", ""),
        &data_size,
        |b, &n| b.iter(|| f3(n)),
    );
    group3.bench_with_input(BenchmarkId::new("View Access", ""), &data_size, |b, &n| {
        b.iter(|| f3_b(n))
    });
    group3.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
