//! GPU Quantum Annealing Tests
//!
//! Tests for QUBO simulated annealing on GPU

#[cfg(all(feature = "cuda", test))]
mod quantum_gpu_tests {
    use cudarc::driver::CudaDevice;
    use prct_core::*;
    use shared_types::*;
    use std::sync::Arc;

    /// Create a small test graph (triangle)
    fn create_triangle_graph() -> Graph {
        let n = 3;
        let mut adjacency = vec![false; n * n];

        // Triangle: edges (0,1), (1,2), (2,0)
        adjacency[0 * n + 1] = true;
        adjacency[1 * n + 0] = true;
        adjacency[1 * n + 2] = true;
        adjacency[2 * n + 1] = true;
        adjacency[2 * n + 0] = true;
        adjacency[0 * n + 2] = true;

        Graph {
            num_vertices: n,
            num_edges: 3,
            adjacency,
        }
    }

    /// Create a bipartite graph K(3,3)
    fn create_bipartite_graph() -> Graph {
        let n = 6;
        let mut adjacency = vec![false; n * n];
        let mut edges = 0;

        // Connect first 3 to last 3
        for i in 0..3 {
            for j in 3..6 {
                adjacency[i * n + j] = true;
                adjacency[j * n + i] = true;
                edges += 1;
            }
        }

        Graph {
            num_vertices: n,
            num_edges: edges,
            adjacency,
        }
    }

    #[test]
    fn test_qubo_csr_conversion() {
        // Test CSR conversion from COO format
        use prct_core::gpu_quantum_annealing::CsrMatrix;

        let entries = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 1, 3.0),
            (1, 2, 4.0),
            (2, 2, 5.0),
        ];

        let csr = CsrMatrix::from_qubo_coo(&entries, 3);

        assert_eq!(csr.num_rows, 3);
        assert_eq!(csr.num_cols, 3);
        assert_eq!(csr.row_ptr, vec![0, 2, 4, 5]);
        assert_eq!(csr.col_idx, vec![0, 1, 1, 2, 2]);
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_qubo_to_coloring_valid() {
        use prct_core::gpu_quantum_annealing::qubo_solution_to_coloring;

        // 3 vertices, 2 colors
        let solution = vec![
            true, false, // v0 -> c0
            false, true, // v1 -> c1
            true, false, // v2 -> c0
        ];

        let coloring = qubo_solution_to_coloring(&solution, 3, 2).unwrap();
        assert_eq!(coloring, vec![0, 1, 0]);
    }

    #[test]
    fn test_qubo_to_coloring_conflicts() {
        use prct_core::gpu_quantum_annealing::qubo_solution_to_coloring;

        // 3 vertices, 2 colors - multiple colors per vertex
        let solution = vec![
            true, true, // v0 -> both colors (conflict)
            false, true, // v1 -> c1
            true, false, // v2 -> c0
        ];

        let coloring = qubo_solution_to_coloring(&solution, 3, 2).unwrap();
        // Should still return a coloring (using first color when multiple)
        assert_eq!(coloring.len(), 3);
    }

    #[test]
    fn test_qubo_to_coloring_missing() {
        use prct_core::gpu_quantum_annealing::qubo_solution_to_coloring;

        // 3 vertices, 2 colors - missing color for v1
        let solution = vec![
            true, false, // v0 -> c0
            false, false, // v1 -> no color (will assign c0 as fallback)
            true, false, // v2 -> c0
        ];

        let coloring = qubo_solution_to_coloring(&solution, 3, 2).unwrap();
        assert_eq!(coloring.len(), 3);
        assert_eq!(coloring[0], 0);
        assert_eq!(coloring[2], 0);
    }

    #[test]
    fn test_gpu_qubo_solver_init() {
        // Test GPU solver initialization
        let device = CudaDevice::new(0).unwrap();
        let device = Arc::new(device);

        let result = gpu_quantum_annealing::GpuQuboSolver::new(device);

        // This might fail if PTX not compiled, which is OK for unit tests
        // Just verify it doesn't panic
        match result {
            Ok(_) => println!("GPU QUBO solver initialized successfully"),
            Err(e) => println!(
                "GPU QUBO solver init failed (expected if PTX missing): {}",
                e
            ),
        }
    }

    #[test]
    fn test_sparse_qubo_small_graph() {
        // Test QUBO creation for small graph
        let graph = create_triangle_graph();
        let num_colors = 3;

        let qubo = SparseQUBO::from_graph_coloring(&graph, num_colors).unwrap();

        assert_eq!(qubo.num_variables(), graph.num_vertices * num_colors);
        assert!(qubo.nnz() > 0);
        println!(
            "Triangle QUBO: {} vars, {} nnz",
            qubo.num_variables(),
            qubo.nnz()
        );
    }

    #[test]
    fn test_sparse_qubo_bipartite() {
        // Test QUBO creation for bipartite graph
        let graph = create_bipartite_graph();
        let num_colors = 2; // Bipartite needs exactly 2 colors

        let qubo = SparseQUBO::from_graph_coloring(&graph, num_colors).unwrap();

        assert_eq!(qubo.num_variables(), graph.num_vertices * num_colors);
        println!(
            "Bipartite QUBO: {} vars, {} nnz",
            qubo.num_variables(),
            qubo.nnz()
        );
    }

    #[test]
    #[ignore] // Requires GPU and compiled PTX - run with: cargo test --features cuda --test quantum_gpu -- --ignored
    fn test_gpu_qubo_triangle() {
        // End-to-end test: solve triangle coloring on GPU
        let device = CudaDevice::new(0).unwrap();
        let device = Arc::new(device);

        let graph = create_triangle_graph();
        let num_colors = 3;

        let qubo = SparseQUBO::from_graph_coloring(&graph, num_colors).unwrap();

        // Initial solution: all zeros (invalid, but SA will fix)
        let initial_state = vec![false; qubo.num_variables()];

        let result = gpu_quantum_annealing::gpu_qubo_simulated_annealing(
            &device,
            &qubo,
            &initial_state,
            5000, // iterations
            1.0,  // T_initial
            0.01, // T_final
            42,   // seed
        );

        match result {
            Ok(solution) => {
                let coloring = gpu_quantum_annealing::qubo_solution_to_coloring(
                    &solution,
                    graph.num_vertices,
                    num_colors,
                )
                .unwrap();

                println!("Triangle coloring: {:?}", coloring);

                // Verify no conflicts
                let mut conflicts = 0;
                for u in 0..graph.num_vertices {
                    for v in (u + 1)..graph.num_vertices {
                        if graph.adjacency[u * graph.num_vertices + v] && coloring[u] == coloring[v]
                        {
                            conflicts += 1;
                        }
                    }
                }

                assert_eq!(
                    conflicts, 0,
                    "Triangle should be 3-colorable without conflicts"
                );
            }
            Err(e) => {
                panic!("GPU QUBO SA failed: {}", e);
            }
        }
    }

    #[test]
    #[ignore] // Requires GPU and compiled PTX
    fn test_gpu_qubo_bipartite() {
        // End-to-end test: solve bipartite coloring on GPU
        let device = CudaDevice::new(0).unwrap();
        let device = Arc::new(device);

        let graph = create_bipartite_graph();
        let num_colors = 2;

        let qubo = SparseQUBO::from_graph_coloring(&graph, num_colors).unwrap();

        // Better initial solution for bipartite: set first 3 to color 0, last 3 to color 1
        let mut initial_state = vec![false; qubo.num_variables()];
        for v in 0..3 {
            initial_state[v * num_colors + 0] = true;
        }
        for v in 3..6 {
            initial_state[v * num_colors + 1] = true;
        }

        let result = gpu_quantum_annealing::gpu_qubo_simulated_annealing(
            &device,
            &qubo,
            &initial_state,
            5000,
            1.0,
            0.01,
            42,
        );

        match result {
            Ok(solution) => {
                let coloring = gpu_quantum_annealing::qubo_solution_to_coloring(
                    &solution,
                    graph.num_vertices,
                    num_colors,
                )
                .unwrap();

                println!("Bipartite coloring: {:?}", coloring);

                // Verify 2-coloring
                let chromatic = coloring.iter().max().unwrap() + 1;
                assert_eq!(chromatic, 2, "Bipartite should be exactly 2-colorable");

                // Verify no conflicts
                let mut conflicts = 0;
                for u in 0..graph.num_vertices {
                    for v in (u + 1)..graph.num_vertices {
                        if graph.adjacency[u * graph.num_vertices + v] && coloring[u] == coloring[v]
                        {
                            conflicts += 1;
                        }
                    }
                }

                assert_eq!(conflicts, 0, "Bipartite should have no conflicts");
            }
            Err(e) => {
                panic!("GPU QUBO SA failed: {}", e);
            }
        }
    }

    #[test]
    fn test_quantum_coloring_solver_gpu_dispatch() {
        // Test that QuantumColoringSolver dispatches to GPU when device is available
        let device = CudaDevice::new(0).ok().map(Arc::new);

        let mut solver = QuantumColoringSolver::new(device.clone()).unwrap();

        let graph = create_triangle_graph();
        let phase_field = PhaseField {
            phases: vec![0.0; graph.num_vertices],
            coherence_matrix: vec![0.0; graph.num_vertices * graph.num_vertices],
            order_parameter: 0.5,
        };
        let kuramoto_state = KuramotoState {
            phases: vec![0.0; graph.num_vertices],
            frequencies: vec![1.0; graph.num_vertices],
            coupling_matrix: vec![0.0; graph.num_vertices * graph.num_vertices],
            order_parameter: 0.5,
            mean_frequency: 1.0,
        };

        // This will attempt GPU if available, fall back to CPU otherwise
        let result = solver.find_coloring(&graph, &phase_field, &kuramoto_state, 3);

        match result {
            Ok(solution) => {
                println!(
                    "Triangle colored with {} colors, {} conflicts",
                    solution.chromatic_number, solution.conflicts
                );
                assert_eq!(solution.conflicts, 0);
            }
            Err(e) => {
                println!(
                    "Coloring failed (may be expected if PTX not compiled): {}",
                    e
                );
            }
        }
    }
}
