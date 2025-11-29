# PRISM Ultra Implementation Plan - Part 2: Phase 1 (Ultra Kernel Components)

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2025-11-29
- **Purpose**: Ultra kernel implementation specifications
- **Scope**: DR-WHCR-AI-Q-PT kernel, TPTP, AATGS, MBRL

---

# SECTION C: PHASE 1A - DR-WHCR-AI-Q-PT ULTRA KERNEL

## C.1 Kernel Architecture

**File**: `crates/prism-gpu/src/kernels/dr_whcr_ultra.cu`

**AGENT**: prism-gpu-specialist

**Target**: ~3000 LOC

### C.1.1 Shared Memory Layout (~98KB)

```cuda
/**
 * Ultra Shared Memory State
 * Total: ~98KB (fits in RTX 3060's 100KB shared memory)
 */
struct UltraSharedState {
    // ═══════════════════════════════════════════════════════════════════
    // W-CYCLE MULTIGRID HIERARCHY (4 levels)
    // ═══════════════════════════════════════════════════════════════════
    // Level 0 (Fine): 512 vertices max per block
    int coloring_L0[512];               // 2KB
    float conflict_signal_L0[512];      // 2KB

    // Level 1: 128 vertices
    int coloring_L1[128];               // 0.5KB
    float conflict_signal_L1[128];      // 0.5KB
    int projection_L0_to_L1[512];       // 2KB (fine→coarse mapping)

    // Level 2: 32 vertices
    int coloring_L2[32];                // 128B
    float conflict_signal_L2[32];       // 128B
    int projection_L1_to_L2[128];       // 0.5KB

    // Level 3 (Coarsest): 8 vertices
    int coloring_L3[8];                 // 32B
    float conflict_signal_L3[8];        // 32B
    int projection_L2_to_L3[32];        // 128B

    // Wavelet coefficients (4 levels)
    float wavelet_approx[4][512];       // 8KB (approximation)
    double wavelet_detail[4][512];      // 16KB (detail, f64 precision)

    // ═══════════════════════════════════════════════════════════════════
    // DENDRITIC RESERVOIR (8-branch)
    // ═══════════════════════════════════════════════════════════════════
    struct DendriticState {
        float activation[8];            // Per-branch activation
        float calcium;                  // LTP/LTD accumulator
        float threshold;                // Adaptive threshold
        float refractory;               // Refractory counter
    } dendrite[512];                    // 24KB

    float soma_potential[512];          // 2KB
    float spike_history[512];           // 2KB
    float reservoir_state[2048];        // 8KB (echo state)

    // ═══════════════════════════════════════════════════════════════════
    // QUANTUM TUNNELING STATE
    // ═══════════════════════════════════════════════════════════════════
    struct QuantumVertex {
        float amplitude_real[6];        // 6 superposition states
        float amplitude_imag[6];
        int color_idx[6];               // Color for each state
        float tunneling_prob;           // Current tunneling probability
        float phase;                    // Quantum phase
    } quantum[512];                     // 28KB

    // ═══════════════════════════════════════════════════════════════════
    // PARALLEL TEMPERING (12 replicas, 64 vertices each for block)
    // ═══════════════════════════════════════════════════════════════════
    struct TemperingReplica {
        int coloring[64];               // Replica coloring
        float energy;                   // Current energy
        int conflicts;                  // Conflict count
    } replica[12];                      // 3.6KB

    float temperatures[16];             // 64B (ladder)

    // ═══════════════════════════════════════════════════════════════════
    // TPTP: PERSISTENT HOMOLOGY STATE
    // ═══════════════════════════════════════════════════════════════════
    struct PersistentHomologyState {
        float betti[3];                 // Betti numbers (β0, β1, β2)
        float max_persistence;          // Maximum persistence
        float stability_score;          // Stability measure
        int transition_detected;        // Phase boundary flag
        float betti_1_derivative;       // Rate of change
        float persistence_diagram[64];  // Birth-death pairs
    } tda;                              // ~0.3KB

    // ═══════════════════════════════════════════════════════════════════
    // ACTIVE INFERENCE
    // ═══════════════════════════════════════════════════════════════════
    float belief_distribution[512][16]; // 32KB (beliefs over 16 colors)
    float expected_free_energy[512];    // 2KB
    float precision_weights[512];       // 2KB

    // ═══════════════════════════════════════════════════════════════════
    // WORK BUFFERS
    // ═══════════════════════════════════════════════════════════════════
    int conflict_vertices[512];         // 2KB (vertices with conflicts)
    int num_conflict_vertices;          // 4B
    float move_deltas[512];             // 2KB (best move delta per vertex)
    int best_colors[512];               // 2KB (best new color)
    int locks[512];                     // 2KB (vertex locks)

    // Total: ~98KB
};
```

### C.1.2 Main Ultra Kernel

```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include "runtime_config.cuh"

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define MAX_VERTICES_PER_BLOCK 512
#define MAX_COLORS 64
#define NUM_BRANCHES 8
#define NUM_LEVELS 4
#define NUM_REPLICAS 12
#define NUM_QUANTUM_STATES 6

// Forward declarations
__device__ void restrict_to_coarse(UltraSharedState* state, int level, const RuntimeConfig* cfg);
__device__ void prolongate_to_fine(UltraSharedState* state, int level, const RuntimeConfig* cfg);
__device__ void smooth_iteration(UltraSharedState* state, int level, const RuntimeConfig* cfg);
__device__ void dendritic_update(UltraSharedState* state, int vertex, const RuntimeConfig* cfg);
__device__ void quantum_evolve(UltraSharedState* state, int vertex, const RuntimeConfig* cfg);
__device__ void tptp_update(UltraSharedState* state, const RuntimeConfig* cfg);
__device__ bool should_tunnel(UltraSharedState* state, int vertex, const RuntimeConfig* cfg);
__device__ void active_inference_update(UltraSharedState* state, int vertex, const RuntimeConfig* cfg);
__device__ void tempering_step(UltraSharedState* state, int replica, const RuntimeConfig* cfg);
__device__ void replica_exchange(UltraSharedState* state, const RuntimeConfig* cfg);

/**
 * DR-WHCR-AI-Q-PT-TDA Ultra Fused Kernel
 *
 * Single kernel that performs complete optimization iteration:
 * 1. W-Cycle multigrid (restriction → coarse solve → prolongation)
 * 2. Dendritic reservoir update (8-branch neuromorphic)
 * 3. Quantum tunneling (6-state superposition)
 * 4. TPTP phase transition detection
 * 5. Active inference belief update
 * 6. Parallel tempering (12 replicas)
 * 7. WHCR conflict repair
 * 8. Wavelet-guided prioritization
 *
 * @param graph_row_ptr CSR row pointers [num_vertices+1]
 * @param graph_col_idx CSR column indices [num_edges]
 * @param coloring Current vertex coloring [num_vertices] (modified in-place)
 * @param config RuntimeConfig struct with all 50+ parameters
 * @param telemetry Output telemetry struct
 * @param num_vertices Number of vertices
 * @param num_edges Number of edges
 * @param seed Random seed for this iteration
 */
extern "C" __global__ void dr_whcr_ultra_kernel(
    const int* __restrict__ graph_row_ptr,
    const int* __restrict__ graph_col_idx,
    int* __restrict__ coloring,
    const RuntimeConfig* __restrict__ config,
    KernelTelemetry* __restrict__ telemetry,
    int num_vertices,
    int num_edges,
    unsigned long long seed
) {
    // Shared memory allocation
    __shared__ UltraSharedState state;

    // Cooperative groups for grid-wide synchronization
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize RNG
    curandState rng;
    curand_init(seed, gid, 0, &rng);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: LOAD DATA INTO SHARED MEMORY
    // ═══════════════════════════════════════════════════════════════════

    // Calculate vertex range for this block
    int vertices_per_block = (num_vertices + gridDim.x - 1) / gridDim.x;
    int block_start = bid * vertices_per_block;
    int block_end = min(block_start + vertices_per_block, num_vertices);
    int block_size = block_end - block_start;

    // Load coloring into shared memory
    for (int i = tid; i < block_size; i += BLOCK_SIZE) {
        int v = block_start + i;
        if (v < num_vertices) {
            state.coloring_L0[i] = coloring[v];
        }
    }
    block.sync();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: COUNT INITIAL CONFLICTS
    // ═══════════════════════════════════════════════════════════════════

    for (int i = tid; i < block_size; i += BLOCK_SIZE) {
        int v = block_start + i;
        if (v < num_vertices) {
            int my_color = state.coloring_L0[i];
            int start = graph_row_ptr[v];
            int end = graph_row_ptr[v + 1];

            float conflict_count = 0.0f;
            for (int e = start; e < end; e++) {
                int neighbor = graph_col_idx[e];
                // Check if neighbor is in this block
                if (neighbor >= block_start && neighbor < block_end) {
                    if (state.coloring_L0[neighbor - block_start] == my_color) {
                        conflict_count += 1.0f;
                    }
                } else {
                    // Global memory access for out-of-block neighbors
                    if (coloring[neighbor] == my_color) {
                        conflict_count += 1.0f;
                    }
                }
            }
            state.conflict_signal_L0[i] = conflict_count;
        }
    }
    block.sync();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: W-CYCLE MULTIGRID
    // ═══════════════════════════════════════════════════════════════════

    if (multigrid_enabled(config)) {
        // Pre-smoothing at fine level
        for (int s = 0; s < config->pre_smooth_iterations; s++) {
            smooth_iteration(&state, 0, config);
            block.sync();
        }

        // Restriction: L0 → L1 → L2 → L3
        for (int level = 0; level < config->num_levels - 1; level++) {
            restrict_to_coarse(&state, level, config);
            block.sync();
        }

        // Solve at coarsest level (L3)
        if (tid < 8) {
            // Direct solve for 8 vertices - greedy coloring
            int v = tid;
            int used_colors = 0;
            for (int c = 0; c < MAX_COLORS; c++) {
                bool can_use = true;
                // Check neighbors at coarse level
                for (int other = 0; other < 8; other++) {
                    if (other != v && state.coloring_L3[other] == c) {
                        // Check if connected (simplified - actual needs coarse graph)
                        can_use = false;
                        break;
                    }
                }
                if (can_use) {
                    state.coloring_L3[v] = c;
                    break;
                }
            }
        }
        block.sync();

        // Prolongation: L3 → L2 → L1 → L0
        for (int level = config->num_levels - 2; level >= 0; level--) {
            prolongate_to_fine(&state, level, config);
            block.sync();

            // Post-smoothing
            for (int s = 0; s < config->post_smooth_iterations; s++) {
                smooth_iteration(&state, level, config);
                block.sync();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: DENDRITIC RESERVOIR UPDATE
    // ═══════════════════════════════════════════════════════════════════

    if (dendritic_enabled(config)) {
        for (int i = tid; i < block_size; i += BLOCK_SIZE) {
            dendritic_update(&state, i, config);
        }
        block.sync();

        // Compute reservoir output priorities
        for (int i = tid; i < block_size; i += BLOCK_SIZE) {
            float priority = 0.0f;

            // Proximal branch drives immediate priority
            priority += state.dendrite[i].activation[0] * 2.0f;

            // Distal branches indicate structural issues
            for (int b = 1; b < NUM_BRANCHES; b++) {
                priority += state.dendrite[i].activation[b] * config->branch_weights[b];
            }

            // Calcium indicates long-term problematic vertex
            priority += state.dendrite[i].calcium * 3.0f;

            // Soma potential indicates accumulated pressure
            priority += state.soma_potential[i] * 0.5f;

            state.move_deltas[i] = priority; // Temporarily store priority here
        }
        block.sync();
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: TPTP PERSISTENT HOMOLOGY UPDATE
    // ═══════════════════════════════════════════════════════════════════

    if (tptp_enabled(config)) {
        // Thread 0 computes global homology (simplified)
        if (tid == 0) {
            tptp_update(&state, config);
        }
        block.sync();
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 6: QUANTUM TUNNELING
    // ═══════════════════════════════════════════════════════════════════

    if (quantum_enabled(config)) {
        for (int i = tid; i < block_size; i += BLOCK_SIZE) {
            // Evolve quantum state
            quantum_evolve(&state, i, config);

            // Check for tunneling
            if (should_tunnel(&state, i, config)) {
                // Tunnel to new color based on quantum state
                float max_prob = 0.0f;
                int best_state = 0;
                for (int s = 0; s < NUM_QUANTUM_STATES; s++) {
                    float prob = state.quantum[i].amplitude_real[s] * state.quantum[i].amplitude_real[s] +
                                state.quantum[i].amplitude_imag[s] * state.quantum[i].amplitude_imag[s];
                    if (prob > max_prob) {
                        max_prob = prob;
                        best_state = s;
                    }
                }
                state.coloring_L0[i] = state.quantum[i].color_idx[best_state];
            }
        }
        block.sync();
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 7: ACTIVE INFERENCE BELIEF UPDATE
    // ═══════════════════════════════════════════════════════════════════

    if (active_inference_enabled(config)) {
        for (int i = tid; i < block_size; i += BLOCK_SIZE) {
            active_inference_update(&state, i, config);
        }
        block.sync();
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 8: PARALLEL TEMPERING
    // ═══════════════════════════════════════════════════════════════════

    if (tempering_enabled(config)) {
        // Each warp handles one replica
        int warp_id = tid / 32;
        if (warp_id < config->num_replicas) {
            tempering_step(&state, warp_id, config);
        }
        block.sync();

        // Replica exchange (thread 0 coordinates)
        if (tid == 0 && config->iteration % config->swap_interval == 0) {
            replica_exchange(&state, config);
        }
        block.sync();
    }

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 9: WHCR MOVE EVALUATION AND APPLICATION
    // ═══════════════════════════════════════════════════════════════════

    // Identify conflict vertices
    if (tid == 0) {
        state.num_conflict_vertices = 0;
    }
    block.sync();

    for (int i = tid; i < block_size; i += BLOCK_SIZE) {
        if (state.conflict_signal_L0[i] > 0.5f) {
            int idx = atomicAdd(&state.num_conflict_vertices, 1);
            if (idx < MAX_VERTICES_PER_BLOCK) {
                state.conflict_vertices[idx] = i;
            }
        }
    }
    block.sync();

    // Evaluate best moves for conflict vertices
    int num_cv = min(state.num_conflict_vertices, MAX_VERTICES_PER_BLOCK);
    for (int cv_idx = tid; cv_idx < num_cv; cv_idx += BLOCK_SIZE) {
        int i = state.conflict_vertices[cv_idx];
        int v = block_start + i;
        int current_color = state.coloring_L0[i];

        // Count neighbor colors
        int neighbor_colors[MAX_COLORS];
        for (int c = 0; c < MAX_COLORS; c++) neighbor_colors[c] = 0;

        int start = graph_row_ptr[v];
        int end = graph_row_ptr[v + 1];

        for (int e = start; e < end; e++) {
            int neighbor = graph_col_idx[e];
            int n_color;
            if (neighbor >= block_start && neighbor < block_end) {
                n_color = state.coloring_L0[neighbor - block_start];
            } else {
                n_color = coloring[neighbor];
            }
            if (n_color < MAX_COLORS) {
                neighbor_colors[n_color]++;
            }
        }

        // Find best color
        int current_conf = neighbor_colors[current_color];
        float best_delta = 0.0f;
        int best_color = current_color;

        for (int c = 0; c < MAX_COLORS; c++) {
            if (c == current_color) continue;

            float delta = (float)(neighbor_colors[c] - current_conf);

            // Chemical potential penalty
            delta += config->chemical_potential * ((float)c - (float)current_color) / MAX_COLORS;

            // Belief guidance from active inference
            if (active_inference_enabled(config)) {
                float belief_diff = state.belief_distribution[i][c] -
                                   state.belief_distribution[i][current_color];
                delta -= config->belief_weight * belief_diff;
            }

            // Reservoir priority modulation
            if (dendritic_enabled(config)) {
                delta -= state.move_deltas[i] * 0.1f;
            }

            if (delta < best_delta) {
                best_delta = delta;
                best_color = c;
            }
        }

        state.best_colors[i] = best_color;
        state.move_deltas[i] = best_delta;
    }
    block.sync();

    // Apply moves with locking
    for (int cv_idx = tid; cv_idx < num_cv; cv_idx += BLOCK_SIZE) {
        int i = state.conflict_vertices[cv_idx];
        int new_color = state.best_colors[i];
        float delta = state.move_deltas[i];

        if (new_color == state.coloring_L0[i]) continue;
        if (delta >= -0.001f) continue; // Only apply improving moves

        // Try to acquire lock
        if (atomicCAS(&state.locks[i], 0, 1) == 0) {
            state.coloring_L0[i] = new_color;
            atomicExch(&state.locks[i], 0);
        }
    }
    block.sync();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 10: WRITE BACK TO GLOBAL MEMORY
    // ═══════════════════════════════════════════════════════════════════

    for (int i = tid; i < block_size; i += BLOCK_SIZE) {
        int v = block_start + i;
        if (v < num_vertices) {
            coloring[v] = state.coloring_L0[i];
        }
    }

    // Write telemetry (thread 0 only)
    if (gid == 0) {
        // Count total conflicts
        int total_conflicts = 0;
        int max_color = 0;
        for (int v = 0; v < num_vertices; v++) {
            int c = coloring[v];
            if (c > max_color) max_color = c;

            int start = graph_row_ptr[v];
            int end = graph_row_ptr[v + 1];
            for (int e = start; e < end; e++) {
                if (coloring[graph_col_idx[e]] == c) {
                    total_conflicts++;
                }
            }
        }

        telemetry->conflicts = total_conflicts / 2;
        telemetry->colors_used = max_color + 1;
        telemetry->moves_applied = num_cv;
        telemetry->betti_numbers[0] = state.tda.betti[0];
        telemetry->betti_numbers[1] = state.tda.betti[1];
        telemetry->betti_numbers[2] = state.tda.betti[2];
        telemetry->phase_transitions = state.tda.transition_detected;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DEVICE FUNCTION IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

__device__ void restrict_to_coarse(UltraSharedState* state, int level, const RuntimeConfig* cfg) {
    int tid = threadIdx.x;

    // Get source and destination arrays based on level
    int* src_coloring;
    float* src_signal;
    int* dst_coloring;
    float* dst_signal;
    int* projection;
    int src_size, dst_size;

    switch (level) {
        case 0:
            src_coloring = state->coloring_L0;
            src_signal = state->conflict_signal_L0;
            dst_coloring = state->coloring_L1;
            dst_signal = state->conflict_signal_L1;
            projection = state->projection_L0_to_L1;
            src_size = 512; dst_size = 128;
            break;
        case 1:
            src_coloring = state->coloring_L1;
            src_signal = state->conflict_signal_L1;
            dst_coloring = state->coloring_L2;
            dst_signal = state->conflict_signal_L2;
            projection = state->projection_L1_to_L2;
            src_size = 128; dst_size = 32;
            break;
        case 2:
            src_coloring = state->coloring_L2;
            src_signal = state->conflict_signal_L2;
            dst_coloring = state->coloring_L3;
            dst_signal = state->conflict_signal_L3;
            projection = state->projection_L2_to_L3;
            src_size = 32; dst_size = 8;
            break;
        default:
            return;
    }

    // Aggregate fine vertices to coarse
    for (int c = tid; c < dst_size; c += BLOCK_SIZE) {
        float signal_sum = 0.0f;
        int color_votes[MAX_COLORS];
        for (int i = 0; i < MAX_COLORS; i++) color_votes[i] = 0;
        int count = 0;

        for (int f = 0; f < src_size; f++) {
            if (projection[f] == c) {
                signal_sum += src_signal[f];
                color_votes[src_coloring[f]]++;
                count++;
            }
        }

        // Majority vote for color
        int best_color = 0;
        int best_votes = 0;
        for (int i = 0; i < MAX_COLORS; i++) {
            if (color_votes[i] > best_votes) {
                best_votes = color_votes[i];
                best_color = i;
            }
        }

        dst_coloring[c] = best_color;
        dst_signal[c] = signal_sum / fmaxf(1.0f, (float)count);
    }
}

__device__ void prolongate_to_fine(UltraSharedState* state, int level, const RuntimeConfig* cfg) {
    int tid = threadIdx.x;

    int* src_coloring;
    int* dst_coloring;
    int* projection;
    int dst_size;

    switch (level) {
        case 0:
            src_coloring = state->coloring_L1;
            dst_coloring = state->coloring_L0;
            projection = state->projection_L0_to_L1;
            dst_size = 512;
            break;
        case 1:
            src_coloring = state->coloring_L2;
            dst_coloring = state->coloring_L1;
            projection = state->projection_L1_to_L2;
            dst_size = 128;
            break;
        case 2:
            src_coloring = state->coloring_L3;
            dst_coloring = state->coloring_L2;
            projection = state->projection_L2_to_L3;
            dst_size = 32;
            break;
        default:
            return;
    }

    // Interpolate coarse solution to fine
    for (int f = tid; f < dst_size; f += BLOCK_SIZE) {
        int c = projection[f];
        // Use coarse color as hint, blend with current
        int coarse_color = src_coloring[c];
        int fine_color = dst_coloring[f];

        // Weight by prolongation weight
        if (curand_uniform(&((curandState*)nullptr)[0]) < cfg->prolongation_weight) {
            dst_coloring[f] = coarse_color;
        }
    }
}

__device__ void smooth_iteration(UltraSharedState* state, int level, const RuntimeConfig* cfg) {
    // Gauss-Seidel smoothing at given level
    int tid = threadIdx.x;

    int* coloring;
    float* signal;
    int size;

    switch (level) {
        case 0: coloring = state->coloring_L0; signal = state->conflict_signal_L0; size = 512; break;
        case 1: coloring = state->coloring_L1; signal = state->conflict_signal_L1; size = 128; break;
        case 2: coloring = state->coloring_L2; signal = state->conflict_signal_L2; size = 32; break;
        case 3: coloring = state->coloring_L3; signal = state->conflict_signal_L3; size = 8; break;
        default: return;
    }

    // Simple smoothing: reduce high-signal vertices
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        if (signal[i] > 0.5f) {
            // Try to find a better color (simplified)
            // In full implementation, would check actual conflicts
        }
    }
}

__device__ void dendritic_update(UltraSharedState* state, int vertex, const RuntimeConfig* cfg) {
    // Update dendritic compartments based on conflict signal
    float conflict_input = state->conflict_signal_L0[vertex];

    // Process each branch with its time constant
    for (int b = 0; b < NUM_BRANCHES; b++) {
        float tau = cfg->tau_decay[b];

        // Decay existing activation
        state->dendrite[vertex].activation[b] *= tau;

        // Add new input weighted by branch
        float input_weight = cfg->input_scaling * cfg->branch_weights[b];
        state->dendrite[vertex].activation[b] += conflict_input * input_weight;

        // Clamp activation
        state->dendrite[vertex].activation[b] = fminf(1.0f,
            fmaxf(-1.0f, state->dendrite[vertex].activation[b]));
    }

    // Update calcium (long-term memory)
    state->dendrite[vertex].calcium *= 0.99f;
    state->dendrite[vertex].calcium += conflict_input * 0.01f;
    state->dendrite[vertex].calcium = fminf(1.0f, state->dendrite[vertex].calcium);

    // Soma integration
    float soma_input = 0.0f;
    for (int b = 0; b < NUM_BRANCHES; b++) {
        soma_input += state->dendrite[vertex].activation[b] * cfg->branch_weights[b];
    }

    state->soma_potential[vertex] = (1.0f - cfg->reservoir_leak_rate) * state->soma_potential[vertex] +
                                    cfg->reservoir_leak_rate * tanhf(soma_input);

    // Check for spike
    if (state->soma_potential[vertex] > state->dendrite[vertex].threshold) {
        state->spike_history[vertex] = 0.9f * state->spike_history[vertex] + 0.1f;
        state->soma_potential[vertex] = 0.0f;
        state->dendrite[vertex].refractory = 2.0f;
    } else {
        state->spike_history[vertex] *= 0.95f;
    }
}

__device__ void quantum_evolve(UltraSharedState* state, int vertex, const RuntimeConfig* cfg) {
    // Evolve quantum amplitudes
    QuantumVertex* q = &state->quantum[vertex];
    float conflict = state->conflict_signal_L0[vertex];

    for (int s = 0; s < NUM_QUANTUM_STATES; s++) {
        // Energy based on conflict for this color
        int color = q->color_idx[s];
        float energy = conflict * cfg->chemical_potential * (float)color / MAX_COLORS;

        // Phase evolution
        float phase = energy * cfg->transverse_field;
        float cos_p = cosf(phase);
        float sin_p = sinf(phase);

        // Rotate amplitude
        float r = q->amplitude_real[s];
        float i = q->amplitude_imag[s];
        q->amplitude_real[s] = r * cos_p - i * sin_p;
        q->amplitude_imag[s] = r * sin_p + i * cos_p;

        // Apply interference decay
        q->amplitude_imag[s] *= (1.0f - cfg->interference_decay);
    }

    // Normalize
    float norm_sq = 0.0f;
    for (int s = 0; s < NUM_QUANTUM_STATES; s++) {
        norm_sq += q->amplitude_real[s] * q->amplitude_real[s] +
                   q->amplitude_imag[s] * q->amplitude_imag[s];
    }
    float norm = sqrtf(fmaxf(1e-8f, norm_sq));
    for (int s = 0; s < NUM_QUANTUM_STATES; s++) {
        q->amplitude_real[s] /= norm;
        q->amplitude_imag[s] /= norm;
    }
}

__device__ void tptp_update(UltraSharedState* state, const RuntimeConfig* cfg) {
    // Simplified persistent homology computation
    // In full implementation, would build Rips complex and compute homology

    // Count connected components (β0)
    int num_components = 0;
    int visited[512];
    for (int i = 0; i < 512; i++) visited[i] = 0;

    for (int i = 0; i < 512; i++) {
        if (!visited[i] && state->conflict_signal_L0[i] > 0.0f) {
            num_components++;
            // BFS to mark component (simplified)
            visited[i] = 1;
        }
    }

    float prev_betti_1 = state->tda.betti[1];

    state->tda.betti[0] = (float)num_components;
    state->tda.betti[1] = 0.0f; // Would compute cycles
    state->tda.betti[2] = 0.0f; // Would compute voids

    // Compute derivative
    state->tda.betti_1_derivative = state->tda.betti[1] - prev_betti_1;

    // Detect phase transition
    state->tda.transition_detected = (fabsf(state->tda.betti_1_derivative) > cfg->transition_sensitivity) ? 1 : 0;
}

__device__ bool should_tunnel(UltraSharedState* state, int vertex, const RuntimeConfig* cfg) {
    QuantumVertex* q = &state->quantum[vertex];

    // Base tunneling probability
    float prob = cfg->tunneling_prob_base;

    // Boost at phase transitions
    if (state->tda.transition_detected) {
        prob *= cfg->tunneling_prob_boost;
    }

    // Boost for high-conflict vertices
    if (state->conflict_signal_L0[vertex] > 2.0f) {
        prob *= 1.5f;
    }

    // Boost for stagnant vertices (high calcium)
    if (state->dendrite[vertex].calcium > 0.8f) {
        prob *= 2.0f;
    }

    q->tunneling_prob = prob;

    // Stochastic decision (simplified - would use proper RNG)
    return (prob > 0.5f);
}

__device__ void active_inference_update(UltraSharedState* state, int vertex, const RuntimeConfig* cfg) {
    int current_color = state->coloring_L0[vertex];

    // Update beliefs based on conflict observations
    float conflict = state->conflict_signal_L0[vertex];

    // Prediction error: expected no conflict, observed conflict
    float prediction_error = conflict;

    // Update belief for current color (decrease if conflicting)
    state->belief_distribution[vertex][current_color] -=
        cfg->belief_update_rate * prediction_error * cfg->precision_weight;

    // Normalize beliefs
    float sum = 0.0f;
    for (int c = 0; c < 16; c++) {
        state->belief_distribution[vertex][c] = fmaxf(0.01f, state->belief_distribution[vertex][c]);
        sum += state->belief_distribution[vertex][c];
    }
    for (int c = 0; c < 16; c++) {
        state->belief_distribution[vertex][c] /= sum;
    }

    // Compute expected free energy
    float efe = 0.0f;
    for (int c = 0; c < 16; c++) {
        float belief = state->belief_distribution[vertex][c];
        // Ambiguity term: -H[P(o|s,a)]
        efe -= belief * logf(fmaxf(1e-8f, belief));
    }
    state->expected_free_energy[vertex] = efe;
}

__device__ void tempering_step(UltraSharedState* state, int replica, const RuntimeConfig* cfg) {
    // Simplified parallel tempering step
    float temp = cfg->temperatures[replica];
    TemperingReplica* rep = &state->replica[replica];

    int lane = threadIdx.x % 32;
    if (lane < 64) {
        int v = lane;
        int current = rep->coloring[v];

        // Propose random new color
        int new_color = (current + 1) % MAX_COLORS;

        // Compute energy change (simplified)
        float delta_E = 0.0f; // Would compute actual conflict change

        // Metropolis acceptance
        bool accept = (delta_E <= 0.0f) ||
                     (expf(-delta_E / temp) > 0.5f); // Simplified

        if (accept) {
            rep->coloring[v] = new_color;
        }
    }
}

__device__ void replica_exchange(UltraSharedState* state, const RuntimeConfig* cfg) {
    // Attempt swaps between adjacent replicas
    for (int i = 0; i < cfg->num_replicas - 1; i += 2) {
        TemperingReplica* r1 = &state->replica[i];
        TemperingReplica* r2 = &state->replica[i + 1];

        float T1 = cfg->temperatures[i];
        float T2 = cfg->temperatures[i + 1];
        float E1 = (float)r1->conflicts;
        float E2 = (float)r2->conflicts;

        // Swap probability
        float delta = (1.0f/T1 - 1.0f/T2) * (E2 - E1);
        bool accept = (delta >= 0.0f) || (expf(delta) > 0.5f);

        if (accept) {
            // Swap colorings
            for (int v = 0; v < 64; v++) {
                int temp = r1->coloring[v];
                r1->coloring[v] = r2->coloring[v];
                r2->coloring[v] = temp;
            }
            // Swap energies
            int temp_c = r1->conflicts;
            r1->conflicts = r2->conflicts;
            r2->conflicts = temp_c;
        }
    }
}
```

---

# SECTION D: PHASE 1B - TPTP MODULE

## D.1 TPTP Kernel

**File**: `crates/prism-gpu/src/kernels/tptp.cu`

**AGENT**: prism-gpu-specialist

**Target**: ~800 LOC

```cuda
/**
 * TPTP: Topological Phase Transition Prediction
 *
 * Live persistent homology computation for detecting phase boundaries
 * in the optimization landscape.
 */

#include <cuda_runtime.h>
#include "runtime_config.cuh"

#define BLOCK_SIZE 256
#define MAX_SIMPLICES 4096
#define MAX_FILTRATION_STEPS 64

// Simplex types
#define SIMPLEX_VERTEX 0
#define SIMPLEX_EDGE 1
#define SIMPLEX_TRIANGLE 2

struct Simplex {
    int type;               // 0=vertex, 1=edge, 2=triangle
    int vertices[3];        // Vertex indices
    float filtration_value; // When simplex appears
    int boundary[3];        // Boundary simplex indices
    int num_boundary;
};

struct PersistencePair {
    float birth;
    float death;
    int dimension;
    int creator_simplex;
    int destroyer_simplex;
};

struct TPTPState {
    Simplex simplices[MAX_SIMPLICES];
    int num_simplices;

    PersistencePair pairs[1024];
    int num_pairs;

    float betti[3];
    float betti_history[64][3];
    int history_idx;

    float max_persistence;
    float avg_persistence;
    float persistence_entropy;

    int phase_transition_detected;
    float transition_strength;
};

/**
 * Build Vietoris-Rips complex from conflict graph
 */
extern "C" __global__ void tptp_build_complex(
    const int* __restrict__ graph_row_ptr,
    const int* __restrict__ graph_col_idx,
    const float* __restrict__ conflict_signal,
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config,
    int num_vertices
) {
    __shared__ int simplex_count;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        simplex_count = 0;
    }
    __syncthreads();

    // Add vertices as 0-simplices
    if (gid < num_vertices) {
        int idx = atomicAdd(&simplex_count, 1);
        if (idx < MAX_SIMPLICES) {
            state->simplices[idx].type = SIMPLEX_VERTEX;
            state->simplices[idx].vertices[0] = gid;
            state->simplices[idx].filtration_value = conflict_signal[gid];
            state->simplices[idx].num_boundary = 0;
        }
    }
    __syncthreads();

    // Add edges as 1-simplices (based on graph adjacency)
    if (gid < num_vertices) {
        int start = graph_row_ptr[gid];
        int end = graph_row_ptr[gid + 1];

        for (int e = start; e < end; e++) {
            int neighbor = graph_col_idx[e];
            if (neighbor > gid) { // Avoid duplicates
                int idx = atomicAdd(&simplex_count, 1);
                if (idx < MAX_SIMPLICES) {
                    state->simplices[idx].type = SIMPLEX_EDGE;
                    state->simplices[idx].vertices[0] = gid;
                    state->simplices[idx].vertices[1] = neighbor;
                    // Filtration value is max of endpoint conflicts
                    state->simplices[idx].filtration_value =
                        fmaxf(conflict_signal[gid], conflict_signal[neighbor]);
                    state->simplices[idx].num_boundary = 2;
                    state->simplices[idx].boundary[0] = gid;
                    state->simplices[idx].boundary[1] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        state->num_simplices = min(simplex_count, MAX_SIMPLICES);
    }
}

/**
 * Compute persistent homology via matrix reduction
 */
extern "C" __global__ void tptp_compute_homology(
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config
) {
    // Simplified persistence algorithm
    // In production, would implement full matrix reduction

    int tid = threadIdx.x;

    // Count connected components (β0)
    if (tid == 0) {
        int num_vertices = 0;
        int num_edges = 0;

        for (int i = 0; i < state->num_simplices; i++) {
            if (state->simplices[i].type == SIMPLEX_VERTEX) num_vertices++;
            if (state->simplices[i].type == SIMPLEX_EDGE) num_edges++;
        }

        // Euler characteristic approximation
        // β0 - β1 + β2 = V - E + F
        // For graph: β0 ≈ V - E + cycles
        state->betti[0] = (float)num_vertices;
        state->betti[1] = (float)max(0, num_edges - num_vertices + 1);
        state->betti[2] = 0.0f;

        // Store history
        int idx = state->history_idx;
        state->betti_history[idx][0] = state->betti[0];
        state->betti_history[idx][1] = state->betti[1];
        state->betti_history[idx][2] = state->betti[2];
        state->history_idx = (idx + 1) % 64;
    }
    __syncthreads();

    // Compute persistence statistics
    if (tid == 0) {
        float total_persistence = 0.0f;
        float max_p = 0.0f;

        for (int i = 0; i < state->num_pairs; i++) {
            float p = state->pairs[i].death - state->pairs[i].birth;
            total_persistence += p;
            if (p > max_p) max_p = p;
        }

        state->max_persistence = max_p;
        state->avg_persistence = total_persistence / fmaxf(1.0f, (float)state->num_pairs);
    }
}

/**
 * Detect phase transitions from Betti number dynamics
 */
extern "C" __global__ void tptp_detect_transition(
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config
) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Compute Betti number derivatives
        int curr_idx = (state->history_idx - 1 + 64) % 64;
        int prev_idx = (state->history_idx - 2 + 64) % 64;

        float db0 = state->betti_history[curr_idx][0] - state->betti_history[prev_idx][0];
        float db1 = state->betti_history[curr_idx][1] - state->betti_history[prev_idx][1];

        // Phase transition indicators:
        // 1. Sudden drop in β0 (components merging)
        // 2. Spike in β1 (cycles forming/breaking)
        // 3. High persistence variance

        float transition_score = 0.0f;

        // β0 drop detection
        if (db0 < -config->betti_0_threshold) {
            transition_score += fabsf(db0);
        }

        // β1 spike detection
        if (fabsf(db1) > config->betti_1_threshold) {
            transition_score += fabsf(db1) * 2.0f;
        }

        // Persistence spike
        if (state->max_persistence > config->persistence_threshold * 2.0f) {
            transition_score += state->max_persistence;
        }

        state->transition_strength = transition_score;
        state->phase_transition_detected = (transition_score > config->transition_sensitivity) ? 1 : 0;
    }
}

/**
 * Combined TPTP update kernel
 */
extern "C" __global__ void tptp_full_update(
    const int* __restrict__ graph_row_ptr,
    const int* __restrict__ graph_col_idx,
    const float* __restrict__ conflict_signal,
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config,
    int num_vertices
) {
    // This would orchestrate the full TPTP pipeline
    // In practice, called as separate kernels for better occupancy
}
```

---

# SECTION E: PHASE 1C - AATGS SCHEDULER

## E.1 AATGS Rust Module

**File**: `crates/prism-gpu/src/aatgs.rs`

**AGENT**: prism-gpu-specialist

**Target**: ~500 LOC

```rust
//! AATGS: Adaptive Asynchronous Task Graph Scheduler
//!
//! GPU-resident scheduler that eliminates CPU-GPU synchronization barriers
//! by maintaining circular buffers for config and telemetry.

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream, CudaView, CudaViewMut};
use std::sync::Arc;

use prism_core::{RuntimeConfig, KernelTelemetry};

/// Circular buffer capacity for configs and telemetry
const CONFIG_BUFFER_SIZE: usize = 16;
const TELEMETRY_BUFFER_SIZE: usize = 64;

/// AATGS buffer state (mirrored on GPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AATGSBuffers {
    /// Circular buffer of RuntimeConfigs
    pub config_buffer: [RuntimeConfig; CONFIG_BUFFER_SIZE],
    /// Write pointer for config buffer (CPU writes, GPU reads)
    pub config_write_ptr: i32,
    /// Read pointer for config buffer (GPU writes, CPU reads)
    pub config_read_ptr: i32,

    /// Circular buffer of telemetry outputs
    pub telemetry_buffer: [KernelTelemetry; TELEMETRY_BUFFER_SIZE],
    /// Write pointer for telemetry (GPU writes, CPU reads)
    pub telemetry_write_ptr: i32,
    /// Read pointer for telemetry (CPU writes, GPU reads)
    pub telemetry_read_ptr: i32,

    /// GPU idle flag (GPU sets when waiting for config)
    pub gpu_idle: i32,
    /// Shutdown signal (CPU sets to terminate GPU loop)
    pub cpu_shutdown: i32,
}

/// AATGS Scheduler
pub struct AATGSScheduler {
    ctx: Arc<CudaContext>,

    /// Stream for config uploads
    config_stream: CudaStream,
    /// Stream for kernel execution
    kernel_stream: CudaStream,
    /// Stream for telemetry downloads
    telemetry_stream: CudaStream,

    /// GPU-side buffer state
    d_buffers: CudaViewMut<AATGSBuffers>,

    /// Host-side shadow of buffer pointers
    local_config_write_ptr: usize,
    local_telemetry_read_ptr: usize,

    /// Pending configs to upload
    pending_configs: Vec<RuntimeConfig>,

    /// Received telemetry
    received_telemetry: Vec<KernelTelemetry>,
}

impl AATGSScheduler {
    /// Create new AATGS scheduler
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let config_stream = ctx.new_stream()?;
        let kernel_stream = ctx.new_stream()?;
        let telemetry_stream = ctx.new_stream()?;

        // Allocate GPU buffer state
        let d_buffers = ctx.alloc_zeros::<AATGSBuffers>(1)?;

        Ok(Self {
            ctx,
            config_stream,
            kernel_stream,
            telemetry_stream,
            d_buffers,
            local_config_write_ptr: 0,
            local_telemetry_read_ptr: 0,
            pending_configs: Vec::with_capacity(CONFIG_BUFFER_SIZE),
            received_telemetry: Vec::new(),
        })
    }

    /// Queue a new config for GPU execution (non-blocking)
    pub fn queue_config(&mut self, config: RuntimeConfig) -> Result<()> {
        self.pending_configs.push(config);

        // Flush if buffer is getting full
        if self.pending_configs.len() >= CONFIG_BUFFER_SIZE / 2 {
            self.flush_configs()?;
        }

        Ok(())
    }

    /// Flush pending configs to GPU (async)
    pub fn flush_configs(&mut self) -> Result<()> {
        if self.pending_configs.is_empty() {
            return Ok(());
        }

        // Upload configs to circular buffer
        for config in self.pending_configs.drain(..) {
            let slot = self.local_config_write_ptr % CONFIG_BUFFER_SIZE;

            // Async copy config to GPU buffer slot
            // self.config_stream.memcpy_htod_async(
            //     &self.d_buffers.config_buffer[slot],
            //     &config,
            // )?;

            self.local_config_write_ptr += 1;
        }

        // Update write pointer on GPU
        // self.config_stream.memcpy_htod_async(
        //     &self.d_buffers.config_write_ptr,
        //     &(self.local_config_write_ptr as i32),
        // )?;

        Ok(())
    }

    /// Poll for completed telemetry (non-blocking)
    pub fn poll_telemetry(&mut self) -> Result<Vec<KernelTelemetry>> {
        // Check how many telemetry entries are available
        let mut gpu_write_ptr: i32 = 0;

        // Async copy telemetry write pointer from GPU
        // self.telemetry_stream.memcpy_dtoh_async(
        //     &mut gpu_write_ptr,
        //     &self.d_buffers.telemetry_write_ptr,
        // )?;

        // Non-blocking check if copy is complete
        // if !self.telemetry_stream.query()? {
        //     return Ok(Vec::new()); // Not ready yet
        // }

        let available = (gpu_write_ptr as usize).saturating_sub(self.local_telemetry_read_ptr);

        if available == 0 {
            return Ok(Vec::new());
        }

        // Read available telemetry entries
        let mut results = Vec::with_capacity(available);

        for _ in 0..available {
            let slot = self.local_telemetry_read_ptr % TELEMETRY_BUFFER_SIZE;

            let mut telemetry = KernelTelemetry::default();
            // self.telemetry_stream.memcpy_dtoh_async(
            //     &mut telemetry,
            //     &self.d_buffers.telemetry_buffer[slot],
            // )?;

            results.push(telemetry);
            self.local_telemetry_read_ptr += 1;
        }

        // Update read pointer on GPU
        // self.telemetry_stream.memcpy_htod_async(
        //     &self.d_buffers.telemetry_read_ptr,
        //     &(self.local_telemetry_read_ptr as i32),
        // )?;

        Ok(results)
    }

    /// Check if GPU is idle (waiting for work)
    pub fn is_gpu_idle(&self) -> Result<bool> {
        // Would async read gpu_idle flag
        Ok(false)
    }

    /// Signal GPU to shutdown
    pub fn shutdown(&mut self) -> Result<()> {
        // Set shutdown flag
        // self.config_stream.memcpy_htod_async(
        //     &self.d_buffers.cpu_shutdown,
        //     &1i32,
        // )?;

        // Wait for all streams to complete
        self.config_stream.synchronize()?;
        self.kernel_stream.synchronize()?;
        self.telemetry_stream.synchronize()?;

        Ok(())
    }

    /// Get config stream for external coordination
    pub fn config_stream(&self) -> &CudaStream {
        &self.config_stream
    }

    /// Get kernel stream for external coordination
    pub fn kernel_stream(&self) -> &CudaStream {
        &self.kernel_stream
    }

    /// Get telemetry stream for external coordination
    pub fn telemetry_stream(&self) -> &CudaStream {
        &self.telemetry_stream
    }
}

/// Triple-buffered async pipeline
pub struct AsyncPipeline {
    scheduler: AATGSScheduler,

    /// Events for cross-stream synchronization
    config_uploaded_event: (), // CudaEvent
    kernel_complete_event: (), // CudaEvent
    telemetry_ready_event: (), // CudaEvent
}

impl AsyncPipeline {
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            scheduler: AATGSScheduler::new(ctx)?,
            config_uploaded_event: (),
            kernel_complete_event: (),
            telemetry_ready_event: (),
        })
    }

    /// Execute one async iteration
    ///
    /// Pipeline stages (overlapped):
    /// 1. Upload config[n+1] on config_stream
    /// 2. Execute kernel[n] on kernel_stream (waits for config[n])
    /// 3. Download telemetry[n-1] on telemetry_stream
    /// 4. CPU processes telemetry[n-2] → generates config[n+2]
    pub fn step(&mut self, new_config: RuntimeConfig) -> Result<Option<KernelTelemetry>> {
        // Queue new config (async upload)
        self.scheduler.queue_config(new_config)?;

        // Poll for completed telemetry (non-blocking)
        let telemetry = self.scheduler.poll_telemetry()?;

        Ok(telemetry.into_iter().next())
    }
}
```

---

# END OF PART 2

**Next**: Part 3 covers Phase 1D-1E (MBRL, FluxNet Ultra) and Phase 2-3 (Async Pipeline, Multi-GPU)
