//! CPU vs GPU Neuromorphic Processing Benchmark (Simplified)
//!
//! Compares compilation/initialization times only since we don't have PTX kernels yet
//!
//! Run with: cargo bench --features cuda --bench cpu_vs_gpu_benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neuromorphic_engine::{reservoir::ReservoirConfig, ReservoirComputer};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use neuromorphic_engine::gpu_reservoir::GpuReservoirComputer;

/// Benchmark: CPU Reservoir Initialization
fn bench_cpu_reservoir_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("reservoir_initialization");

    for size in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(BenchmarkId::new("CPU", size), size, |b, &size| {
            b.iter(|| {
                black_box(ReservoirComputer::new(
                    size, // reservoir_size
                    50,   // input_size
                    0.95, // spectral_radius
                    0.3,  // connection_prob
                    0.3,  // leak_rate
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark: GPU Reservoir Initialization
#[cfg(feature = "cuda")]
fn bench_gpu_reservoir_init(c: &mut Criterion) {
    if let Ok(device) = CudaDevice::new(0) {
        let mut group = c.benchmark_group("reservoir_initialization");

        for size in [100, 500, 1000, 2000].iter() {
            group.bench_with_input(BenchmarkId::new("GPU", size), size, |b, &size| {
                let config = ReservoirConfig {
                    size,
                    input_size: 50,
                    spectral_radius: 0.95,
                    input_scaling: 0.1,
                    connection_prob: 0.3,
                    leak_rate: 0.3,
                    noise_level: 0.01,
                    enable_plasticity: false,
                    stdp_profile: neuromorphic_engine::STDPProfile::Balanced,
                };

                b.iter(|| {
                    black_box(GpuReservoirComputer::new_shared(
                        config.clone(),
                        device.clone(),
                    ))
                });
            });
        }

        group.finish();
    } else {
        println!("GPU not available, skipping GPU benchmarks");
    }
}

/// Benchmark: GPU Memory Throughput
#[cfg(feature = "cuda")]
fn bench_gpu_memory_throughput(c: &mut Criterion) {
    if let Ok(device) = CudaDevice::new(0) {
        let mut group = c.benchmark_group("memory_throughput");

        for size_mb in [1, 10, 100].iter() {
            let num_elements = size_mb * 1024 * 1024 / 4; // f32 = 4 bytes

            group.bench_with_input(BenchmarkId::new("GPU_HtoD", size_mb), size_mb, |b, _| {
                let data: Vec<f32> = vec![1.0; num_elements];

                b.iter(|| {
                    let _ = black_box(device.htod_sync_copy(&data));
                });
            });
        }

        group.finish();
    }
}

// Configure criterion benchmark groups
criterion_group!(cpu_benches, bench_cpu_reservoir_init,);

#[cfg(feature = "cuda")]
criterion_group!(
    gpu_benches,
    bench_gpu_reservoir_init,
    bench_gpu_memory_throughput,
);

#[cfg(feature = "cuda")]
criterion_main!(cpu_benches, gpu_benches);

#[cfg(not(feature = "cuda"))]
criterion_main!(cpu_benches);
