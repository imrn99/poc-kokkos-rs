use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::{
    routines::{
        parallel_for,
        parameters::{ExecutionPolicy, ExecutionSpace, RangePolicy, Schedule},
    },
    view::{parameters::Layout, ViewOwned},
};

// this bench is used to evaluate the efficiency of our parallel_for
// routine compared to regular loops.

// 1D regular for init & populating
fn f1(length: usize) {
    let mut v_y = ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
    black_box(&mut v_y); // prevents the first init to be optimized away
    (0..500).for_each(|_| {
        (0..length).for_each(|i| {
            v_y[[i]] = 1.0;
        });
        black_box(&v_y);
    })
}

// 1D parallel_for (serial) init & populating
fn f1_b(length: usize) {
    let mut v_y = ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
    black_box(&mut v_y); // prevents the first init to be optimized away
    let execp = ExecutionPolicy {
        space: ExecutionSpace::Serial,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    (0..500).for_each(|_| {
        let execp_loc = execp.clone();
        parallel_for::<0, 1, _>(execp_loc, |[i]| v_y[[i]] = 1.0).unwrap();
        black_box(&v_y);
    })
}

// 3D regular for init & populating
fn f2(length: usize) {
    let mut v_y = ViewOwned::new_from_data(
        vec![0.0; length * length * length],
        Layout::Right,
        [length; 3],
    );
    black_box(&mut v_y); // prevents the first init to be optimized away

    (0..length).for_each(|i| {
        (0..length).for_each(|j| {
            (0..length).for_each(|k| {
                v_y[[i, j, k]] = 1.0;
            })
        })
    });
    black_box(&v_y);
}

// 3D parallel_for (serial) init & populating
fn f2_b(length: usize) {
    let mut v_y = ViewOwned::new_from_data(
        vec![0.0; length * length * length],
        Layout::Right,
        [length; 3],
    );
    black_box(&mut v_y); // prevents the first init to be optimized away
    let execp = ExecutionPolicy {
        space: ExecutionSpace::Serial,
        range: RangePolicy::MDRangePolicy([0..length, 0..length, 0..length]),
        schedule: Schedule::Static,
    };

    parallel_for::<0, 3, _>(execp, |indices| v_y[indices] = 1.0).unwrap();
    black_box(&v_y);
}

// 5D regular for init & populating
fn f3(length: usize) {
    let mut v_y = ViewOwned::new_from_data(
        vec![0.0; length * length * length * length * length],
        Layout::Right,
        [length; 5],
    );
    black_box(&mut v_y); // prevents the first init to be optimized away

    (0..length).for_each(|i| {
        (0..length).for_each(|j| {
            (0..length).for_each(|k| {
                (0..length).for_each(|l| {
                    (0..length).for_each(|m| {
                        v_y[[i, j, k, l, m]] = 1.0;
                    })
                })
            })
        })
    });
    black_box(&v_y);
}

// 5D parallel_for (serial) init & populating
fn f3_b(length: usize) {
    let mut v_y = ViewOwned::new_from_data(
        vec![0.0; length * length * length * length * length],
        Layout::Right,
        [length; 5],
    );
    black_box(&mut v_y); // prevents the first init to be optimized away
    let execp = ExecutionPolicy {
        space: ExecutionSpace::Serial,
        range: RangePolicy::MDRangePolicy([0..length, 0..length, 0..length, 0..length, 0..length]),
        schedule: Schedule::Static,
    };

    parallel_for::<0, 5, _>(execp, |indices| v_y[indices] = 1.0).unwrap();
    black_box(&v_y);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 11; // 2048 length vector, 2048*2048 matrix
    let mut length = 2_usize.pow(DATA_SIZE);

    let mut group1 = c.benchmark_group("1D init & populate");
    group1.bench_with_input(
        BenchmarkId::new("Regular for (serial)", ""),
        &(length),
        |b, n| b.iter(|| f1(*n)),
    );
    group1.bench_with_input(
        BenchmarkId::new("Parallel for (serial)", ""),
        &(length),
        |b, n| b.iter(|| f1_b(*n)),
    );
    group1.finish();

    length = 2_usize.pow(8);

    let mut group2 = c.benchmark_group("3D init & populate");
    group2.bench_with_input(
        BenchmarkId::new("Regular for (serial)", ""),
        &(length),
        |b, n| b.iter(|| f2(*n)),
    );
    group2.bench_with_input(
        BenchmarkId::new("Parallel for (serial)", ""),
        &(length),
        |b, n| b.iter(|| f2_b(*n)),
    );
    group2.finish();

    length = 2_usize.pow(5);

    let mut group3 = c.benchmark_group("5D init & populate");
    group3.bench_with_input(
        BenchmarkId::new("Regular for (serial)", ""),
        &(length),
        |b, n| b.iter(|| f3(*n)),
    );
    group3.bench_with_input(
        BenchmarkId::new("Parallel for (serial)", ""),
        &(length),
        |b, n| b.iter(|| f3_b(*n)),
    );
    group3.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
