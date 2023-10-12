use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Currently a partial gemv
// y = Ax / y = xA
// instead of
// y = s1*Au + s2*v

// regular matrix-vector product
fn f1(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    // in this case, we can use Rust's iterator directly to easily operate
    // line by line.
    let y: Vec<f64> = A
        .chunks(length)
        .map(|row| row.iter().zip(x.iter()).map(|(r_i, x_i)| r_i * x_i).sum())
        .collect();
    black_box(y);
}

// regular matrix-vector product; using indexes
fn f1_b(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    // As a reference, an implementation using indexes
    let mut y: Vec<f64> = vec![0.0; length];
    // col and row indexes of the matrix
    for row in 0..length {
        for col in 0..length {
            // using unchecked accesses to keep the comparison "fair"
            // as iterators bypass those
            unsafe {
                *y.get_unchecked_mut(row) +=
                    A.get_unchecked(row * length + col) * x.get_unchecked(col)
            }
        }
    }
    black_box(y);
}

// regular vector-matrix product
fn f2(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    // in the case of a vector-matrix product, the "row-first" layout (i.e. 2D LayoutRight)
    // does not allow us to make use of Rust's iterators -> back to indexes
    let mut y: Vec<f64> = vec![0.0; length];
    // col and row indexes of the matrix
    for col in 0..length {
        for row in 0..length {
            // using unchecked accesses to keep the comparison "fair"
            // as iterators bypass those
            unsafe {
                *y.get_unchecked_mut(col) +=
                    x.get_unchecked(row) * A.get_unchecked(row * length + col)
            }
        }
    }
    black_box(y);
}

// vector-matrix product with an adapted layout
fn f3(size: u32) {
    let length = 2_usize.pow(size);
    let x = vec![1.0; length];
    #[allow(non_snake_case)]
    let A = vec![1.0; length * length];
    // Thanks to the "row first" layout (i.e. 2D LayoutLeft), we can use
    // the iterators again
    // The code is essentially the same as the matrix-vector product
    let y: Vec<f64> = A
        .chunks(length)
        .map(|col| x.iter().zip(col.iter()).map(|(x_i, c_i)| x_i * c_i).sum())
        .collect();
    black_box(y);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    let data_size: u32 = 11; // 2048 length vector, 2048*2048 matrix

    let mut group = c.benchmark_group("gemv");
    group.bench_with_input(
        BenchmarkId::new("Matrix-Vector Product (iterators)", ""),
        &data_size,
        |b, &n| b.iter(|| f1(n)),
    );
    group.bench_with_input(
        BenchmarkId::new("Matrix-Vector Product (indexes)", ""),
        &data_size,
        |b, &n| b.iter(|| f1_b(n)),
    );
    group.bench_with_input(
        BenchmarkId::new("Vector-Matrix Product (indexes)", ""),
        &data_size,
        |b, &n| b.iter(|| f2(n)),
    );
    group.bench_with_input(
        BenchmarkId::new("Vector-Matrix Product w/ adapted layout (iterators)", ""),
        &data_size,
        |b, &n| b.iter(|| f3(n)),
    );
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
