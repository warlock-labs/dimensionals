use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dimensionals::{Dimensional, LinearArrayStorage};

// TODO: This needs meaningful benchmarks for common operations useful in
// quantitive situations

fn bench_dimensional_array_creation_zeros(c: &mut Criterion) {
    let shape = [1000, 1000];
    c.bench_function("dimensional_array_creation_zeros", |b| {
        b.iter(|| Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::zeros(shape))
    });
}

fn bench_dimensional_array_creation_ones(c: &mut Criterion) {
    let shape = [1000, 1000];
    c.bench_function("dimensional_array_creation_ones", |b| {
        b.iter(|| Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::ones(shape))
    });
}

fn bench_dimensional_array_indexing(c: &mut Criterion) {
    let shape = [1000, 1000];
    let array = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::zeros(shape);

    c.bench_function("dimensional_array_indexing", |b| {
        b.iter(|| {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    black_box(array[[i, j]]);
                }
            }
        })
    });
}

fn bench_dimensional_array_mutable_indexing(c: &mut Criterion) {
    let shape = [1000, 1000];
    let mut array = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::zeros(shape);

    c.bench_function("dimensional_array_mutable_indexing", |b| {
        b.iter(|| {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    array[[i, j]] = 1.0;
                }
            }
        })
    });
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    let shape1 = [100, 200];
    let shape2 = [200, 100];
    let m1 = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::ones(shape1);
    let m2 = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::ones(shape2);

    c.bench_function("matrix_multiplication", |b| {
        b.iter(|| m1.mat_mul(&m2))
    });
}

fn bench_matrix_transpose(c: &mut Criterion) {
    let shape = [1000, 1000];
    let m = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::ones(shape);

    c.bench_function("matrix_transpose", |b| {
        b.iter(|| m.transpose())
    });
}

fn bench_matrix_trace(c: &mut Criterion) {
    let shape = [1000, 1000];
    let m = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::ones(shape);

    c.bench_function("matrix_trace", |b| {
        b.iter(|| m.trace())
    });
}

criterion_group!(
    benches,
    bench_dimensional_array_creation_zeros,
    bench_dimensional_array_creation_ones,
    bench_dimensional_array_indexing,
    bench_dimensional_array_mutable_indexing,
    bench_matrix_multiplication,
    bench_matrix_transpose,
    bench_matrix_trace
);
criterion_main!(benches);
