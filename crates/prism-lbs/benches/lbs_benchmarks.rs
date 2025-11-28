use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| 1u64.wrapping_mul(1)));
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
