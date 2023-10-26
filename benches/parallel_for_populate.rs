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

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 11; // 2048 length vector, 2048*2048 matrix
    let length = 2_usize.pow(DATA_SIZE);

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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
