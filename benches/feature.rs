use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poc_kokkos_rs::{
    functor::KernelArgs,
    routines::{
        parallel_for,
        parameters::{ExecutionPolicy, ExecutionSpace, RangePolicy, Schedule},
    },
    view::{parameters::Layout, ViewOwned},
};

// this bench is used to assess whether the parallel_for routines
// switches backend accordingly to feature. It should be executed
// multiple time by the user, each time with a different feature

// 1D regular for init & populating
fn f1(length: usize) {
    let mut v_y = ViewOwned::new_from_data(vec![0.0; length], Layout::Right, [length]);
    black_box(&mut v_y); // prevents the first init to be optimized away
    let execp = ExecutionPolicy {
        space: ExecutionSpace::DeviceCPU,
        range: RangePolicy::RangePolicy(0..length),
        schedule: Schedule::Static,
    };

    let kernel = |arg: KernelArgs<1>| match arg {
        KernelArgs::Index1D(i) => {
            v_y.set([i], 1.0);
            black_box(&v_y[[i]]);
        }
        KernelArgs::IndexND(_) => unimplemented!(),
        KernelArgs::Handle => unimplemented!(),
    };
    parallel_for::<0, 1>(execp, kernel).unwrap();
    black_box(&v_y);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Generate/Define the input
    const DATA_SIZE: u32 = 20;
    let length = 2_usize.pow(DATA_SIZE);

    let mut group = c.benchmark_group("parallel_for");
    group.bench_with_input(
        BenchmarkId::new("feature-specific time", ""),
        &(length),
        |b, n| b.iter(|| f1(*n)),
    );
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
