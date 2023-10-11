use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::view::{parameters::Layout, ViewBase, ViewOwned};

// regular matrix vector product
fn f1(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![vec![1.0; length]; length];
    let y: Vec<f64> = A
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(f1, f2)| f1 * f2).sum())
        .collect();
    black_box(y);
}

// regular matrix vector product, with a flattened matrix
fn f1_b(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    let y: Vec<f64> = A
        .chunks_exact(length)
        .map(|row| row.iter().zip(x.iter()).map(|(f1, f2)| f1 * f2).sum())
        .collect();
    black_box(y);
}

// views vector matrix product
fn f2(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    // in the case of vector-matrix product, we would like to iterate on the
    // column of the matrix; We can't do that efficiently with the regular
    // "row first" layout (i.e. a LayoutRight)
    let v_x: ViewOwned<'_, 1, f64> = ViewBase::new_from_data(x, Layout::Right, [length]);
    #[allow(non_snake_case)]
    let v_A: ViewOwned<'_, 2, f64> = ViewBase::new_from_data(A, Layout::Right, [length, length]);
    // result
    let y: ViewOwned<'_, 1, f64> = ViewBase::new(Layout::Right, [length]);
}

// views vector matrix product with adapted layout
fn f3(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    // in the case of vector-matrix product, we would like to iterate on the
    // column of the matrix; We can't do that efficiently with the regular
    // "row first" layout (i.e. a LayoutRight)
    let v_x: ViewOwned<'_, 1, f64> = ViewBase::new_from_data(x, Layout::Right, [length]);
    #[allow(non_snake_case)]
    let v_A: ViewOwned<'_, 2, f64> = ViewBase::new_from_data(A, Layout::Left, [length, length]);
    // result
    let y: ViewOwned<'_, 1, f64> = ViewBase::new(Layout::Right, [length]);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    let data_size: u32 = 11; // 2048 length vector, 2048*2048 matrix

    let mut group = c.benchmark_group("gemv");
    group.bench_with_input(
        BenchmarkId::new("Regular Matrix-Vector Product", "number of iterations"),
        &data_size,
        |b, &n| b.iter(|| f1(n)),
    );
    group.bench_with_input(
        BenchmarkId::new(
            "Regular Matrix-Vector Product (Flattened Matrix)",
            "number of iterations",
        ),
        &data_size,
        |b, &n| b.iter(|| f1_b(n)),
    );

    group.bench_with_input(
        BenchmarkId::new("Views Vector-Matrix Product", "number of iterations"),
        &data_size,
        |b, &n| b.iter(|| f2(n)),
    );
    group.bench_with_input(
        BenchmarkId::new(
            "Views Vector-Matrix Product with Adapted Layout ",
            "number of iterations",
        ),
        &data_size,
        |b, &n| b.iter(|| f3(n)),
    );
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
