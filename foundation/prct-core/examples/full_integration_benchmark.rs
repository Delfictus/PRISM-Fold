//! Full Integration Cascading Pipeline Benchmark
//!
//! Tests the complete Full Integration pipeline (Option C + Advanced Features)
//! on DSJC1000.5 benchmark.
//!
//! Expected Performance: 562 (baseline) → 95-100 colors
//! World Record: 83 colors
//! Target Gap: ~1.15-1.20x world record
//!
//! Run with:
//! cargo run --release --features cuda --example full_integration_benchmark -- ../../benchmarks/dimacs/DSJC1000.5.col

use prct_core::{dimacs_parser::parse_graph_file, CascadingPipeline};
use shared_types::KuramotoState;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get benchmark file from command line or use default
    let args: Vec<String> = env::args().collect();
    let benchmark_file = if args.len() > 1 {
        args[1].clone()
    } else {
        "../../benchmarks/dimacs/DSJC1000.5.col".to_string()
    };

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         FULL INTEGRATION CASCADING PIPELINE TEST          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("Benchmark file: {}", benchmark_file);
    println!();

    // Parse graph
    println!("Loading graph...");
    let load_start = Instant::now();
    let graph = parse_graph_file(&benchmark_file)?;
    println!(
        "✅ Loaded: {} vertices, {} edges ({:.2}ms)",
        graph.num_vertices,
        graph.num_edges,
        load_start.elapsed().as_micros() as f64 / 1000.0
    );

    // Calculate graph properties
    let avg_degree = (2 * graph.num_edges) as f64 / graph.num_vertices as f64;
    let density =
        (2 * graph.num_edges) as f64 / (graph.num_vertices * (graph.num_vertices - 1)) as f64;
    println!("Average degree: {:.2}", avg_degree);
    println!("Graph density: {:.4}", density);
    println!();

    // Create initial Kuramoto state
    // For this standalone test, we'll create a simple random phase initialization
    let n = graph.num_vertices;
    let phases = (0..n)
        .map(|i| (i as f64 * 2.0 * std::f64::consts::PI) / n as f64)
        .collect();
    let initial_kuramoto = KuramotoState {
        phases,
        natural_frequencies: vec![1.0; n],
        coupling_matrix: vec![0.0; n * n],
        order_parameter: 0.0,
        mean_phase: 0.0,
    };

    // Run Full Integration Pipeline
    println!("Starting Full Integration Cascading Pipeline...");
    let total_start = Instant::now();

    let mut pipeline = CascadingPipeline::new();
    let result = pipeline.optimize(&graph, &initial_kuramoto)?;

    let total_time = total_start.elapsed().as_secs_f64();

    // Print final results
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                     FINAL RESULTS                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("Colors achieved: {} colors", result.chromatic_number);
    println!("Conflicts: {}", result.conflicts);
    println!("Total time: {:.2}s", total_time);
    println!();

    // Comparison with world record and targets
    let world_record = 83;
    let gap_to_wr = result.chromatic_number as f64 / world_record as f64;

    println!("Performance Analysis:");
    println!("  World Record: {} colors", world_record);
    println!("  Our Result:   {} colors", result.chromatic_number);
    println!("  Gap to WR:    {:.2}x", gap_to_wr);

    if result.chromatic_number <= 100 {
        println!("  Status: ✅ TARGET ACHIEVED (<100 colors)");
    } else if result.chromatic_number <= 110 {
        println!("  Status: ⚡ EXCELLENT (100-110 colors)");
    } else if result.chromatic_number <= 120 {
        println!("  Status: ✓ GOOD (110-120 colors)");
    } else {
        println!("  Status: ⚠️  NEEDS IMPROVEMENT (>120 colors)");
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                  BENCHMARK COMPLETE                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
