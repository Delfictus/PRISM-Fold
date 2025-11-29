//! GPU-Accelerated Graph Coloring Example
//!
//! Demonstrates PRCT pipeline with neuromorphic GPU acceleration
//!
//! Run with: cargo run --features cuda --example gpu_graph_coloring

#[cfg(feature = "cuda")]
use prct_core::adapters::NeuromorphicAdapter;
use prct_core::{
    adapters::{CouplingAdapter, QuantumAdapter},
    ports::{NeuromorphicEncodingParams, NeuromorphicPort},
};
use shared_types::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== GPU-Accelerated PRCT Graph Coloring ===\n");

    // 1. Create test graph (10-vertex wheel graph)
    println!("1. Creating test graph...");
    let graph = create_wheel_graph(10);
    println!(
        "   ✅ Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    // 2. Initialize adapters
    println!("\n2. Initializing PRCT adapters...");

    #[cfg(feature = "cuda")]
    {
        match cudarc::driver::CudaContext::new(0) {
            Ok(device) => {
                println!("   ✅ GPU detected");

                let neuro_adapter = NeuromorphicAdapter::new(device.clone())?;
                let quantum_adapter = QuantumAdapter::new(Some(device))?;
                let coupling_adapter = CouplingAdapter::new(0.5)?;

                println!("   ✅ Adapters initialized (GPU mode)");

                // 3. Run DRPP pipeline
                println!("\n3. Running DRPP algorithm with GPU acceleration...");
                run_drpp_pipeline(graph, neuro_adapter, quantum_adapter, coupling_adapter)?;
            }
            Err(e) => {
                println!("   ⚠️  GPU not available: {}", e);
                println!("   Falling back to CPU mode...");
                run_cpu_pipeline(graph)?;
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("   ℹ️  Running in CPU mode (cuda feature not enabled)");
        run_cpu_pipeline(graph)?;
    }

    println!("\n=== Test Complete ===\n");
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_drpp_pipeline(
    graph: Graph,
    neuro: NeuromorphicAdapter,
    quantum: QuantumAdapter,
    coupling: CouplingAdapter,
) -> Result<(), Box<dyn std::error::Error>> {
    use prct_core::ports::{PhysicsCouplingPort, QuantumPort};
    use std::time::Instant;

    let start = Instant::now();

    // Encode graph as spike pattern (GPU)
    let params = NeuromorphicEncodingParams::default();
    let spike_pattern = neuro.encode_graph_as_spikes(&graph, &params)?;
    println!(
        "   ✅ Spike encoding: {} spikes in {:.2}ms",
        spike_pattern.spikes.len(),
        start.elapsed().as_micros() as f64 / 1000.0
    );

    // Process through neuromorphic adapter (GPU)
    let neuro_state = neuro.process_and_detect_patterns(&spike_pattern)?;
    println!(
        "   ✅ Neuromorphic processing: {} activations",
        neuro_state.neuron_states.len()
    );

    // Quantum evolution
    let evolution_params = EvolutionParams {
        dt: 0.01,
        strength: 1.0,
        damping: 0.1,
        temperature: 300.0, // Room temperature in Kelvin
    };
    let hamiltonian = quantum.build_hamiltonian(&graph, &evolution_params)?;

    // Create initial quantum state
    let initial_state = QuantumState {
        amplitudes: vec![(1.0, 0.0); graph.num_vertices],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    let quantum_state = quantum.evolve_state(&hamiltonian, &initial_state, 1.0)?;
    println!(
        "   ✅ Quantum evolution: {} amplitudes",
        quantum_state.amplitudes.len()
    );

    // Coupling analysis
    let coupling_result = coupling.get_bidirectional_coupling(&neuro_state, &quantum_state)?;
    println!(
        "   ✅ Coupling: order parameter = {:.4}",
        coupling_result.kuramoto_state.order_parameter
    );

    let total_time = start.elapsed();
    println!(
        "\n   ⏱️  Total pipeline time: {:.2}ms",
        total_time.as_micros() as f64 / 1000.0
    );

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_cpu_pipeline(graph: Graph) -> Result<(), Box<dyn std::error::Error>> {
    println!("   ⚠️  CPU-only mode not yet implemented");
    println!(
        "   Graph has {} vertices and {} edges",
        graph.num_vertices, graph.num_edges
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cpu_pipeline(graph: Graph) -> Result<(), Box<dyn std::error::Error>> {
    println!("   ⚠️  CPU fallback not yet implemented");
    println!(
        "   Graph has {} vertices and {} edges",
        graph.num_vertices, graph.num_edges
    );
    Ok(())
}

/// Create a wheel graph for testing
fn create_wheel_graph(n: usize) -> Graph {
    let num_vertices = n + 1; // n rim vertices + 1 hub
    let num_edges = 2 * n; // n spokes + n rim edges

    let mut adjacency = vec![false; num_vertices * num_vertices];
    let mut edges = Vec::new();

    // Hub (vertex 0) connects to all rim vertices
    for i in 1..num_vertices {
        adjacency[0 * num_vertices + i] = true;
        adjacency[i * num_vertices + 0] = true;
        edges.push((0, i, 1.0));
        edges.push((i, 0, 1.0));
    }

    // Rim vertices form a cycle
    for i in 1..num_vertices {
        let next = if i == num_vertices - 1 { 1 } else { i + 1 };
        adjacency[i * num_vertices + next] = true;
        adjacency[next * num_vertices + i] = true;
        edges.push((i, next, 1.0));
        edges.push((next, i, 1.0));
    }

    Graph {
        num_vertices,
        num_edges,
        adjacency,
        edges,
        coordinates: None,
    }
}
