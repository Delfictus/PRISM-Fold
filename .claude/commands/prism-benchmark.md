# PRISM Benchmark Agent

You are a specialized agent for implementing and running comprehensive benchmarks for PRISM.

## Mission
Implement industry-standard benchmarks and testing infrastructure:
1. **Graph Coloring**: DIMACS benchmark suite
2. **LBS/Biomolecular**: PDBBind, DUD-E, Astex Diverse Set
3. **GPU Performance**: Kernel timing, memory profiling
4. **Integration Tests**: End-to-end pipeline verification

## Target Metrics

| Benchmark | Metric | Target |
|-----------|--------|--------|
| DSJC500.5 | Colors | ≤48 |
| DSJC1000.5 | Colors | ≤83 |
| PDBBind Core | DCC < 4Å | ≥70% |
| DUD-E | Enrichment AUC | ≥0.75 |
| GPU Utilization | SM Occupancy | ≥80% |
| Latency (10K atoms) | End-to-end | <500ms |
| Memory | Peak VRAM | <4GB |

---

## 1. DIMACS Graph Coloring Benchmarks

File: `crates/prism-cli/benches/dimacs.rs`
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_core::Graph;
use prism_pipeline::{PipelineConfig, PipelineOrchestrator};
use std::path::Path;

const DIMACS_DIR: &str = "data/dimacs";

struct DimacsResult {
    graph_name: String,
    vertices: usize,
    edges: usize,
    colors: usize,
    conflicts: usize,
    time_ms: f64,
}

fn load_dimacs_graph(name: &str) -> Graph {
    let path = Path::new(DIMACS_DIR).join(format!("{}.col", name));
    Graph::from_dimacs(&path).expect("Failed to load DIMACS graph")
}

fn benchmark_dimacs_suite(c: &mut Criterion) {
    let graphs = vec![
        ("DSJC125.5", 48),   // Known chromatic bound
        ("DSJC250.5", 28),
        ("DSJC500.5", 48),
        ("DSJC1000.5", 83),
        ("flat300_28_0", 28),
        ("le450_15c", 15),
        ("le450_25c", 25),
    ];

    let mut group = c.benchmark_group("DIMACS");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    for (name, target_colors) in graphs {
        let graph = load_dimacs_graph(name);

        group.bench_with_input(
            BenchmarkId::new(name, graph.num_vertices()),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let config = PipelineConfig::default();
                    let mut orchestrator = PipelineOrchestrator::new(config).unwrap();
                    orchestrator.run(graph.clone())
                });
            },
        );
    }

    group.finish();
}

fn verify_dimacs_quality() {
    let benchmarks = vec![
        ("DSJC125.5", 48),
        ("DSJC250.5", 28),
        ("DSJC500.5", 48),
        ("DSJC1000.5", 83),
    ];

    println!("\n{:=<60}", "");
    println!("DIMACS Quality Verification");
    println!("{:=<60}", "");
    println!("{:<20} {:>8} {:>8} {:>10} {:>10}",
             "Graph", "Target", "Actual", "Conflicts", "Status");
    println!("{:-<60}", "");

    for (name, target) in benchmarks {
        let graph = load_dimacs_graph(name);
        let config = PipelineConfig::default();
        let mut orchestrator = PipelineOrchestrator::new(config).unwrap();
        let result = orchestrator.run(graph).unwrap();

        let status = if result.colors <= target && result.conflicts == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        };

        println!("{:<20} {:>8} {:>8} {:>10} {:>10}",
                 name, target, result.colors, result.conflicts, status);
    }

    println!("{:=<60}\n", "");
}

criterion_group!(benches, benchmark_dimacs_suite);
criterion_main!(benches);
```

---

## 2. LBS/Biomolecular Benchmarks

File: `crates/prism-lbs/benches/industry.rs`
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_lbs::{PocketDetector, LbsConfig, Protein};
use std::path::Path;

// PDBBind Core Set: 285 high-quality protein-ligand complexes
fn pdbbind_benchmark(c: &mut Criterion) {
    let pdbbind_dir = Path::new("benchmarks/pdbbind_core");

    if !pdbbind_dir.exists() {
        eprintln!("PDBBind data not found at {:?}", pdbbind_dir);
        return;
    }

    let complexes: Vec<_> = std::fs::read_dir(pdbbind_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "pdb").unwrap_or(false))
        .take(20)  // Subset for CI
        .collect();

    let mut group = c.benchmark_group("PDBBind-Core");
    group.sample_size(10);

    let config = LbsConfig::default();
    let detector = PocketDetector::new(&config).unwrap();

    for entry in complexes {
        let path = entry.path();
        let pdb_id = path.file_stem().unwrap().to_str().unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(pdb_id),
            &path,
            |b, path| {
                let protein = Protein::from_pdb(path).unwrap();
                b.iter(|| detector.detect_pockets(&protein));
            },
        );
    }

    group.finish();
}

// DUD-E: Directory of Useful Decoys - Enhanced
fn dud_e_benchmark(c: &mut Criterion) {
    let dud_e_dir = Path::new("benchmarks/dud_e");

    if !dud_e_dir.exists() {
        eprintln!("DUD-E data not found at {:?}", dud_e_dir);
        return;
    }

    let mut group = c.benchmark_group("DUD-E");

    // Sample targets
    let targets = vec!["aa2ar", "abl1", "ace", "aces", "ada"];

    for target in targets {
        let receptor_path = dud_e_dir.join(target).join("receptor.pdb");
        if !receptor_path.exists() { continue; }

        let config = LbsConfig::default();
        let detector = PocketDetector::new(&config).unwrap();
        let protein = Protein::from_pdb(&receptor_path).unwrap();

        group.bench_function(target, |b| {
            b.iter(|| detector.detect_pockets(&protein));
        });
    }

    group.finish();
}

/// LBS accuracy metrics
pub fn compute_lbs_metrics(
    predicted: &[Pocket],
    ground_truth: &BindingSite,
) -> LbsMetrics {
    // Find best matching pocket
    let best_pocket = predicted.iter()
        .min_by(|a, b| {
            let dist_a = a.center.distance_to(&ground_truth.center);
            let dist_b = b.center.distance_to(&ground_truth.center);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

    let dcc = best_pocket
        .map(|p| p.center.distance_to(&ground_truth.center))
        .unwrap_or(f32::INFINITY);

    let success = dcc < 4.0;  // Standard threshold

    // Volume overlap computation
    let volumetric_overlap = best_pocket
        .map(|p| compute_volume_overlap(p, ground_truth))
        .unwrap_or(0.0);

    LbsMetrics {
        dcc,
        success_rate: if success { 1.0 } else { 0.0 },
        volumetric_overlap,
        discretized_overlap: compute_discretized_overlap(best_pocket, ground_truth),
        enrichment_auc: 0.0,  // Computed separately
    }
}

#[derive(Debug)]
pub struct LbsMetrics {
    pub dcc: f32,              // Distance Center to Center (Å)
    pub success_rate: f32,     // 1.0 if DCC < 4Å
    pub volumetric_overlap: f32,
    pub discretized_overlap: f32,
    pub enrichment_auc: f32,
}

criterion_group!(benches, pdbbind_benchmark, dud_e_benchmark);
criterion_main!(benches);
```

---

## 3. GPU Performance Benchmarks

File: `crates/prism-gpu/benches/kernels.rs`
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_gpu::{CudaContext, WhcrGpu};
use std::time::Instant;

fn benchmark_whcr_kernels(c: &mut Criterion) {
    let ctx = CudaContext::new().expect("CUDA required for GPU benchmarks");

    let sizes = vec![1000, 5000, 10000, 50000];

    let mut group = c.benchmark_group("WHCR-Kernels");

    for size in sizes {
        // Generate random graph
        let graph = generate_random_graph(size, 0.5);
        let mut whcr = WhcrGpu::new(ctx.device(), &graph).unwrap();

        group.bench_with_input(
            BenchmarkId::new("count_conflicts", size),
            &size,
            |b, _| {
                b.iter(|| whcr.count_conflicts());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("evaluate_moves", size),
            &size,
            |b, _| {
                b.iter(|| whcr.evaluate_moves(64));  // 64 colors
            },
        );

        group.bench_with_input(
            BenchmarkId::new("apply_moves", size),
            &size,
            |b, _| {
                b.iter(|| whcr.apply_moves());
            },
        );
    }

    group.finish();
}

fn benchmark_sasa_kernel(c: &mut Criterion) {
    let ctx = CudaContext::new().expect("CUDA required");

    let atom_counts = vec![1000, 5000, 10000, 25000];

    let mut group = c.benchmark_group("SASA-Kernel");
    group.sample_size(10);

    for count in atom_counts {
        let atoms = generate_random_atoms(count);

        group.bench_with_input(
            BenchmarkId::new("spatial_grid", count),
            &atoms,
            |b, atoms| {
                b.iter(|| compute_sasa_with_grid(&ctx, atoms, 1.4, 92));
            },
        );
    }

    group.finish();
}

fn benchmark_memory_bandwidth(c: &mut Criterion) {
    let ctx = CudaContext::new().expect("CUDA required");

    let sizes_mb = vec![1, 10, 100, 500, 1000];

    let mut group = c.benchmark_group("Memory-Bandwidth");

    for size_mb in sizes_mb {
        let size_bytes = size_mb * 1024 * 1024;
        let data: Vec<f32> = vec![0.0; size_bytes / 4];

        group.bench_with_input(
            BenchmarkId::new("htod_copy", size_mb),
            &data,
            |b, data| {
                b.iter(|| ctx.device().htod_copy(data.clone()));
            },
        );
    }

    group.finish();
}

/// GPU telemetry collection
pub fn collect_gpu_metrics() -> GpuMetrics {
    use nvml_wrapper::Nvml;

    let nvml = Nvml::init().expect("NVML required");
    let device = nvml.device_by_index(0).expect("GPU 0 required");

    GpuMetrics {
        utilization: device.utilization_rates().unwrap().gpu as f32 / 100.0,
        memory_used: device.memory_info().unwrap().used as f64 / 1e9,
        memory_total: device.memory_info().unwrap().total as f64 / 1e9,
        temperature: device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu).unwrap() as f32,
        power_draw: device.power_usage().unwrap() as f32 / 1000.0,
        sm_clock: device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM).unwrap(),
    }
}

#[derive(Debug)]
pub struct GpuMetrics {
    pub utilization: f32,     // 0.0-1.0
    pub memory_used: f64,     // GB
    pub memory_total: f64,    // GB
    pub temperature: f32,     // Celsius
    pub power_draw: f32,      // Watts
    pub sm_clock: u32,        // MHz
}

criterion_group!(benches, benchmark_whcr_kernels, benchmark_sasa_kernel, benchmark_memory_bandwidth);
criterion_main!(benches);
```

---

## 4. Integration Tests

File: `tests/integration/pipeline_e2e.rs`
```rust
use prism_pipeline::{PipelineConfig, PipelineOrchestrator};
use prism_core::Graph;
use std::path::Path;

#[test]
fn test_full_pipeline_dsjc250() {
    let graph = Graph::from_dimacs(Path::new("data/dimacs/DSJC250.5.col")).unwrap();

    let config = PipelineConfig {
        max_colors: 50,
        max_iterations: 10000,
        gpu_enabled: true,
        ..Default::default()
    };

    let mut orchestrator = PipelineOrchestrator::new(config).unwrap();
    let result = orchestrator.run(graph).unwrap();

    assert_eq!(result.conflicts, 0, "Pipeline must produce zero conflicts");
    assert!(result.colors <= 30, "DSJC250.5 should achieve ≤30 colors");
}

#[test]
fn test_gpu_fallback() {
    // Test that pipeline works even without GPU
    std::env::set_var("CUDA_VISIBLE_DEVICES", "");

    let graph = Graph::from_dimacs(Path::new("data/dimacs/DSJC125.5.col")).unwrap();
    let config = PipelineConfig::default();

    let mut orchestrator = PipelineOrchestrator::new(config).unwrap();
    let result = orchestrator.run(graph).unwrap();

    assert_eq!(result.conflicts, 0);
}

#[test]
fn test_all_phases_execute() {
    let graph = Graph::from_dimacs(Path::new("data/dimacs/DSJC125.5.col")).unwrap();
    let config = PipelineConfig::default();

    let mut orchestrator = PipelineOrchestrator::new(config).unwrap();
    let result = orchestrator.run_with_telemetry(graph).unwrap();

    // Verify all phases executed
    assert!(result.phase_timings.len() >= 7, "All 7 phases should execute");
    for (phase_idx, timing) in result.phase_timings.iter().enumerate() {
        assert!(timing.duration_ms > 0.0, "Phase {} should have non-zero duration", phase_idx);
    }
}

#[test]
fn test_fluxnet_integration() {
    let graph = Graph::from_dimacs(Path::new("data/dimacs/DSJC125.5.col")).unwrap();

    let config = PipelineConfig {
        use_fluxnet: true,
        fluxnet_epsilon: 0.1,
        ..Default::default()
    };

    let mut orchestrator = PipelineOrchestrator::new(config).unwrap();
    let result = orchestrator.run(graph).unwrap();

    assert_eq!(result.conflicts, 0);
    assert!(result.fluxnet_updates > 0, "FluxNet should have made updates");
}
```

---

## Files to Create
1. `crates/prism-cli/benches/dimacs.rs`
2. `crates/prism-lbs/benches/industry.rs`
3. `crates/prism-gpu/benches/kernels.rs`
4. `tests/integration/pipeline_e2e.rs`
5. `tests/integration/mod.rs`

## Validation
```bash
# Run all benchmarks
cargo bench --all

# Run specific benchmark suite
cargo bench --package prism-cli -- dimacs
cargo bench --package prism-lbs -- pdbbind

# Run integration tests
cargo test --test pipeline_e2e

# Generate benchmark report
cargo bench -- --save-baseline main
```

## Output Format
Report:
1. DIMACS results (graph, colors, conflicts, time)
2. LBS accuracy metrics (DCC, success rate)
3. GPU kernel timings
4. Memory usage profile
5. All tests passing
