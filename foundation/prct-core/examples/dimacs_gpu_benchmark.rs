//! DIMACS Benchmark with GPU-Accelerated PRCT
//!
//! Tests PRCT pipeline on standard graph coloring benchmarks
//!
//! Run with: cargo run --features cuda --example dimacs_gpu_benchmark -- <benchmark_file>
//! Example: cargo run --features cuda --example dimacs_gpu_benchmark -- ../../benchmarks/dimacs/myciel6.col

use prct_core::{
    coloring::phase_guided_coloring,
    dimacs_parser::parse_graph_file,
    ports::{NeuromorphicEncodingParams, NeuromorphicPort, PhysicsCouplingPort, QuantumPort},
    QuantumColoringSolver,
};

#[cfg(feature = "cuda")]
use prct_core::adapters::{CouplingAdapter, NeuromorphicAdapter, QuantumAdapter};

use shared_types::{EvolutionParams, Graph, QuantumState};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get benchmark file from command line or use default
    let args: Vec<String> = env::args().collect();
    let benchmark_file = if args.len() > 1 {
        args[1].clone()
    } else {
        "../../benchmarks/dimacs/myciel6.col".to_string()
    };

    println!("\n=== PRCT GPU DIMACS Benchmark ===\n");
    println!("Benchmark file: {}", benchmark_file);

    // Parse graph (auto-detects DIMACS or MTX format)
    println!("\n1. Loading graph...");
    let load_start = Instant::now();
    let graph = parse_graph_file(&benchmark_file)?;
    println!(
        "   ✅ Loaded: {} vertices, {} edges ({:.2}ms)",
        graph.num_vertices,
        graph.num_edges,
        load_start.elapsed().as_micros() as f64 / 1000.0
    );

    // Calculate graph properties
    let avg_degree = (2 * graph.num_edges) as f64 / graph.num_vertices as f64;
    let density =
        (2 * graph.num_edges) as f64 / (graph.num_vertices * (graph.num_vertices - 1)) as f64;
    println!("   Average degree: {:.2}", avg_degree);
    println!("   Graph density: {:.4}", density);

    // Initialize GPU adapters
    println!("\n2. Initializing GPU adapters...");
    let init_start = Instant::now();

    #[cfg(feature = "cuda")]
    {
        match cudarc::driver::CudaDevice::new(0) {
            Ok(device) => {
                println!("   ✅ GPU detected");

                let neuro_adapter = NeuromorphicAdapter::new(device.clone())?;
                let quantum_adapter = QuantumAdapter::new(Some(device))?;
                let coupling_adapter = CouplingAdapter::new(0.5)?;

                println!(
                    "   ✅ Adapters initialized ({:.2}ms)",
                    init_start.elapsed().as_micros() as f64 / 1000.0
                );

                // Run PRCT pipeline
                run_prct_pipeline(graph, neuro_adapter, quantum_adapter, coupling_adapter)?;
            }
            Err(e) => {
                println!("   ⚠️  GPU not available: {}", e);
                println!("   This benchmark requires GPU acceleration");
                return Err(Box::new(e));
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("   ⚠️  CUDA feature not enabled");
        println!("   Run with: cargo run --features cuda --example dimacs_gpu_benchmark");
        return Ok(());
    }

    println!("\n=== Benchmark Complete ===\n");
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_prct_pipeline(
    graph: Graph,
    neuro: NeuromorphicAdapter,
    quantum: QuantumAdapter,
    coupling: CouplingAdapter,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Running PRCT Pipeline with GPU Acceleration...");
    println!("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let pipeline_start = Instant::now();
    let mut phase_times = Vec::new();

    // Phase 1: Neuromorphic Spike Encoding
    println!("\n   Phase 1: Neuromorphic Encoding (GPU)");
    let phase_start = Instant::now();
    let params = NeuromorphicEncodingParams::default();
    let spike_pattern = neuro.encode_graph_as_spikes(&graph, &params)?;
    let phase_time = phase_start.elapsed();
    phase_times.push(("Spike Encoding", phase_time));

    println!("      Spikes generated: {}", spike_pattern.spikes.len());
    println!(
        "      Time: {:.3}ms",
        phase_time.as_micros() as f64 / 1000.0
    );
    println!(
        "      Throughput: {:.0} spikes/ms",
        spike_pattern.spikes.len() as f64 / (phase_time.as_micros() as f64 / 1000.0)
    );

    // Phase 2: Reservoir Processing
    println!("\n   Phase 2: Reservoir Computing (GPU)");
    let phase_start = Instant::now();
    let neuro_state = neuro.process_and_detect_patterns(&spike_pattern)?;
    let phase_time = phase_start.elapsed();
    phase_times.push(("Reservoir Processing", phase_time));

    println!(
        "      Neuron activations: {}",
        neuro_state.neuron_states.len()
    );
    println!(
        "      Time: {:.3}ms",
        phase_time.as_micros() as f64 / 1000.0
    );

    // Phase 3: Quantum Evolution
    println!("\n   Phase 3: Quantum Hamiltonian Evolution");
    let phase_start = Instant::now();

    let evolution_params = EvolutionParams {
        dt: 0.01,
        strength: 1.0,
        damping: 0.1,
        temperature: 300.0,
    };

    let hamiltonian = quantum.build_hamiltonian(&graph, &evolution_params)?;

    let initial_state = QuantumState {
        amplitudes: vec![(1.0, 0.0); graph.num_vertices],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    let quantum_state = quantum.evolve_state(&hamiltonian, &initial_state, 1.0)?;
    let phase_time = phase_start.elapsed();
    phase_times.push(("Quantum Evolution", phase_time));

    println!(
        "      Quantum amplitudes: {}",
        quantum_state.amplitudes.len()
    );
    println!(
        "      Phase coherence: {:.4}",
        quantum_state.phase_coherence
    );
    println!("      Energy: {:.4}", quantum_state.energy);
    println!(
        "      Time: {:.3}ms",
        phase_time.as_micros() as f64 / 1000.0
    );

    // Phase 4: Bidirectional Coupling
    println!("\n   Phase 4: Neuromorphic-Quantum Coupling");
    let phase_start = Instant::now();
    let coupling_result = coupling.get_bidirectional_coupling(&neuro_state, &quantum_state)?;
    let phase_time = phase_start.elapsed();
    phase_times.push(("Coupling Analysis", phase_time));

    println!(
        "      Kuramoto order parameter: {:.4}",
        coupling_result.kuramoto_state.order_parameter
    );
    println!(
        "      Mean phase: {:.4} rad",
        coupling_result.kuramoto_state.mean_phase
    );
    println!(
        "      Coupling quality: {:.4}",
        coupling_result.coupling_quality
    );
    println!(
        "      Time: {:.3}ms",
        phase_time.as_micros() as f64 / 1000.0
    );

    // Transfer entropy analysis
    println!("\n   Phase 5: Information Flow Analysis");
    println!(
        "      Neuro → Quantum: {:.4} bits",
        coupling_result.neuro_to_quantum_entropy.entropy_bits
    );
    println!(
        "      Quantum → Neuro: {:.4} bits",
        coupling_result.quantum_to_neuro_entropy.entropy_bits
    );
    println!(
        "      Confidence: {:.2}%",
        coupling_result.neuro_to_quantum_entropy.confidence * 100.0
    );

    // Phase 6: Extract Graph Coloring (Greedy Baseline)
    println!("\n   Phase 6: Phase-Guided Graph Coloring (Greedy Baseline)");
    let coloring_start = Instant::now();

    let phase_field = quantum.get_phase_field(&quantum_state)?;

    // Try different target colors (estimate from graph structure)
    let estimated_chromatic = estimate_chromatic_number(&graph);
    let target_colors = estimated_chromatic + 10; // Give some slack

    let greedy_solution = match phase_guided_coloring(
        &graph,
        &phase_field,
        &coupling_result.kuramoto_state,
        target_colors,
    ) {
        Ok(solution) => {
            let coloring_time = coloring_start.elapsed();
            phase_times.push(("Graph Coloring (Greedy)", coloring_time));

            println!("      Colors used: {}", solution.chromatic_number);
            println!("      Conflicts: {}", solution.conflicts);
            println!("      Quality score: {:.4}", solution.quality_score);
            println!(
                "      Time: {:.3}ms",
                coloring_time.as_micros() as f64 / 1000.0
            );

            if solution.conflicts == 0 {
                println!("      ✅ VALID COLORING FOUND!");
            } else {
                println!("      ⚠️  Coloring has conflicts (may need more colors)");
            }
            Some(solution)
        }
        Err(e) => {
            println!("      ⚠️  Coloring extraction failed: {}", e);
            println!("      (Phase field may need parameter tuning)");
            None
        }
    };

    // Phase 7: Quantum Annealing Optimization (NEW!)
    if greedy_solution.is_some() && graph.num_vertices <= 1000 {
        println!("\n   Phase 7: Quantum Annealing Optimization (EXPERIMENTAL)");
        let qa_start = Instant::now();

        #[cfg(feature = "cuda")]
        let device = cudarc::driver::CudaDevice::new(0).ok();
        #[cfg(not(feature = "cuda"))]
        let device: Option<std::sync::Arc<()>> = None;

        match QuantumColoringSolver::new(
            #[cfg(feature = "cuda")]
            device,
        ) {
            Ok(mut qa_solver) => {
                match qa_solver.find_coloring(
                    &graph,
                    &phase_field,
                    &coupling_result.kuramoto_state,
                    estimated_chromatic,
                ) {
                    Ok(qa_solution) => {
                        let qa_time = qa_start.elapsed();
                        phase_times.push(("Quantum Annealing", qa_time));

                        let greedy_colors = greedy_solution.as_ref().unwrap().chromatic_number;
                        let improvement = if greedy_colors > qa_solution.chromatic_number {
                            let ratio = greedy_colors as f64 / qa_solution.chromatic_number as f64;
                            format!("{:.2}x better", ratio)
                        } else if greedy_colors == qa_solution.chromatic_number {
                            "same".to_string()
                        } else {
                            format!(
                                "{:.2}x worse",
                                qa_solution.chromatic_number as f64 / greedy_colors as f64
                            )
                        };

                        println!(
                            "      Colors used: {} (greedy: {})",
                            qa_solution.chromatic_number, greedy_colors
                        );
                        println!("      Improvement: {}", improvement);
                        println!("      Conflicts: {}", qa_solution.conflicts);
                        println!("      Quality score: {:.4}", qa_solution.quality_score);
                        println!("      Time: {:.3}ms", qa_time.as_micros() as f64 / 1000.0);

                        if qa_solution.chromatic_number < greedy_colors {
                            println!("      ✅ QUANTUM ANNEALING IMPROVED SOLUTION!");
                        } else {
                            println!(
                                "      🟡 Quantum annealing did not improve (may need more steps)"
                            );
                        }
                    }
                    Err(e) => {
                        println!("      ⚠️  Quantum annealing failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("      ⚠️  Could not initialize quantum solver: {}", e);
            }
        }
    } else if graph.num_vertices > 1000 {
        println!("\n   Phase 7: Quantum Annealing (skipped for large graph > 1000 vertices)");
    }

    // Pipeline summary
    let total_time = pipeline_start.elapsed();
    println!("\n   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n4. Performance Summary");
    println!("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for (phase_name, phase_time) in &phase_times {
        let percentage = (phase_time.as_micros() as f64 / total_time.as_micros() as f64) * 100.0;
        println!(
            "      {:25} {:8.3}ms  ({:5.1}%)",
            phase_name,
            phase_time.as_micros() as f64 / 1000.0,
            percentage
        );
    }

    println!("   ─────────────────────────────────────────────────");
    println!(
        "      {:25} {:8.3}ms  (100.0%)",
        "TOTAL",
        total_time.as_micros() as f64 / 1000.0
    );

    // Performance metrics
    println!("\n5. Graph Complexity Metrics");
    println!("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let vertices_per_ms = graph.num_vertices as f64 / (total_time.as_micros() as f64 / 1000.0);
    let edges_per_ms = graph.num_edges as f64 / (total_time.as_micros() as f64 / 1000.0);

    println!(
        "      Vertices processed: {} ({:.0} vertices/ms)",
        graph.num_vertices, vertices_per_ms
    );
    println!(
        "      Edges processed: {} ({:.0} edges/ms)",
        graph.num_edges, edges_per_ms
    );

    // Coupling strength interpretation
    println!("\n6. Coupling Strength Analysis");
    println!("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let order = coupling_result.kuramoto_state.order_parameter;
    let coupling_level = if order > 0.9 {
        "VERY STRONG"
    } else if order > 0.7 {
        "STRONG"
    } else if order > 0.5 {
        "MODERATE"
    } else if order > 0.3 {
        "WEAK"
    } else {
        "VERY WEAK"
    };

    println!("      Order parameter: {:.4} → {}", order, coupling_level);
    println!("      Synchronization: {:.1}%", order * 100.0);

    if order > 0.7 {
        println!("      ✅ Strong neuromorphic-quantum coupling achieved");
        println!("      ✅ System is coherent and well-synchronized");
    } else if order > 0.5 {
        println!("      🟡 Moderate coupling - system partially synchronized");
    } else {
        println!("      ⚠️  Weak coupling - may need parameter tuning");
    }

    Ok(())
}

/// Estimate chromatic number from graph structure
/// Uses max degree + 1 as an upper bound heuristic
#[cfg(feature = "cuda")]
fn estimate_chromatic_number(graph: &Graph) -> usize {
    let n = graph.num_vertices;
    let mut max_degree = 0;

    for i in 0..n {
        let degree = graph.adjacency[i * n..(i + 1) * n]
            .iter()
            .filter(|&&e| e)
            .count();
        max_degree = max_degree.max(degree);
    }

    // Brooks' theorem upper bound: max_degree + 1 (unless complete or odd cycle)
    (max_degree + 1).max(3)
}
