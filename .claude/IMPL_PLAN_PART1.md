# PRISM Ultra Implementation Plan - Part 1: Architecture & Phase 0

## Document Metadata
- **Version**: 1.0.0
- **Created**: 2025-11-29
- **Purpose**: Agent-referenceable implementation specification
- **Scope**: cudarc migration, architecture overview, foundational infrastructure

---

# SECTION A: SYSTEM ARCHITECTURE OVERVIEW

## A.1 Target Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PRISM ULTRA ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         CPU LAYER (Rust)                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │   FluxNet   │  │    MBRL     │  │  Telemetry  │  │   Config    │     │    │
│  │  │ Controller  │  │ World Model │  │  Processor  │  │   Manager   │     │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │    │
│  │         │                │                │                │            │    │
│  │         └────────────────┴────────────────┴────────────────┘            │    │
│  │                                   │                                      │    │
│  │                    ┌──────────────▼──────────────┐                      │    │
│  │                    │     AATGS Orchestrator      │                      │    │
│  │                    │  (Async Task Graph Sched)   │                      │    │
│  │                    └──────────────┬──────────────┘                      │    │
│  └───────────────────────────────────┼──────────────────────────────────────┘    │
│                                      │                                           │
│  ┌───────────────────────────────────┼──────────────────────────────────────┐    │
│  │                         FFI LAYER (cudarc 0.18+)                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │  Stream A   │  │  Stream B   │  │  Stream C   │  │  Stream D   │     │    │
│  │  │ (Config Up) │  │  (Kernel)   │  │ (Telem Dn)  │  │  (P2P Xfer) │     │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │    │
│  │         │                │                │                │            │    │
│  │         └────────────────┴────────────────┴────────────────┘            │    │
│  │                                   │                                      │    │
│  │                    ┌──────────────▼──────────────┐                      │    │
│  │                    │   RuntimeConfig Struct      │                      │    │
│  │                    │   (50+ params, single xfer) │                      │    │
│  │                    └──────────────┬──────────────┘                      │    │
│  └───────────────────────────────────┼──────────────────────────────────────┘    │
│                                      │                                           │
│  ┌───────────────────────────────────┼──────────────────────────────────────┐    │
│  │                         GPU LAYER (CUDA)                                 │    │
│  │                                   │                                      │    │
│  │         ┌─────────────────────────▼─────────────────────────┐           │    │
│  │         │        DR-WHCR-AI-Q-PT-TDA ULTRA KERNEL           │           │    │
│  │         │  ┌─────────────────────────────────────────────┐  │           │    │
│  │         │  │              SHARED MEMORY (~98KB)          │  │           │    │
│  │         │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │           │    │
│  │         │  │  │W-Cycle  │ │Dendritic│ │ Quantum │       │  │           │    │
│  │         │  │  │Hierarchy│ │Reservoir│ │Tunneling│       │  │           │    │
│  │         │  │  │4 Levels │ │8 Branch │ │6 States │       │  │           │    │
│  │         │  │  └─────────┘ └─────────┘ └─────────┘       │  │           │    │
│  │         │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │           │    │
│  │         │  │  │ Active  │ │Parallel │ │  TPTP   │       │  │           │    │
│  │         │  │  │Inference│ │Tempering│ │Homology │       │  │           │    │
│  │         │  │  │ Beliefs │ │12 Replic│ │ Betti   │       │  │           │    │
│  │         │  │  └─────────┘ └─────────┘ └─────────┘       │  │           │    │
│  │         │  └─────────────────────────────────────────────┘  │           │    │
│  │         └───────────────────────────────────────────────────┘           │    │
│  │                                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │     │    │
│  │  │  (Primary)  │◄─┼─► (P2P)    ◄─┼─►  (P2P)    ◄─┼─►  (P2P)     │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## A.2 Component Inventory

| Component | Location | LOC Target | Agent Owner |
|-----------|----------|------------|-------------|
| cudarc Migration | `crates/prism-gpu/src/*.rs` | ~2000 modified | prism-gpu-specialist |
| Ultra Kernel CUDA | `crates/prism-gpu/src/kernels/dr_whcr_ultra.cu` | ~3000 new | prism-gpu-specialist |
| TPTP Module | `crates/prism-gpu/src/kernels/tptp.cu` | ~800 new | prism-gpu-specialist |
| AATGS Scheduler | `crates/prism-gpu/src/aatgs.rs` | ~500 new | prism-gpu-specialist |
| MBRL World Model | `crates/prism-fluxnet/src/mbrl.rs` | ~600 new | prism-architect |
| Stream Manager | `crates/prism-gpu/src/stream_manager.rs` | ~400 new | prism-gpu-specialist |
| Multi-GPU Pool | `crates/prism-gpu/src/multi_device_pool.rs` | ~600 modified | prism-gpu-specialist |
| RuntimeConfig | `crates/prism-core/src/runtime_config.rs` | ~300 new | prism-architect |
| FluxNet Ultra | `crates/prism-fluxnet/src/ultra_controller.rs` | ~500 new | prism-architect |

## A.3 Dependency Graph

```
RuntimeConfig ─────┐
                   │
AATGS ─────────────┼──► Ultra Kernel ──► PTX
                   │         │
Stream Manager ────┘         │
       │                     │
       ▼                     ▼
Multi-GPU Pool ◄──── cudarc 0.18+ FFI
       │
       ▼
FluxNet Ultra ◄──── MBRL World Model
       │
       ▼
Telemetry ──► Benchmarks
```

---

# SECTION B: PHASE 0 - CUDARC 0.18+ MIGRATION

## B.1 Migration Overview

**Objective**: Upgrade from cudarc 0.9 to 0.18+ to unlock async operations, streams, and struct-based parameter passing.

**Breaking Changes Summary**:
| cudarc 0.9 | cudarc 0.18+ | Migration Action |
|------------|--------------|------------------|
| `CudaDevice` | `CudaContext` | Find/replace + type updates |
| `device.htod_sync_copy()` | `stream.memcpy_htod()` | Refactor to stream-based |
| `device.dtoh_sync_copy()` | `stream.memcpy_dtoh()` | Refactor to stream-based |
| `func.launch()` | `stream.launch()` | Add stream parameter |
| `CudaSlice<T>` | `CudaView<T>` / `CudaViewMut<T>` | Update buffer types |
| 12-param limit | Struct params | Implement RuntimeConfig struct |

## B.2 File-by-File Migration Instructions

### B.2.1 `crates/prism-gpu/Cargo.toml`

**Current**:
```toml
cudarc = { version = "0.9", features = ["std"] }
```

**Target**:
```toml
cudarc = { version = "0.18", features = ["std", "driver", "nvrtc"] }
```

### B.2.2 `crates/prism-gpu/src/context.rs`

**AGENT**: prism-gpu-specialist

**Changes Required**:

1. **Import Updates**:
```rust
// OLD
use cudarc::driver::{CudaDevice, CudaSlice, CudaFunction, LaunchConfig, LaunchAsync};

// NEW
use cudarc::driver::{
    CudaContext, CudaStream, CudaModule, CudaFunction,
    CudaView, CudaViewMut, DevicePtr, DevicePtrMut,
    LaunchConfig, result::DriverError,
};
```

2. **Context Creation**:
```rust
// OLD
let device = CudaDevice::new(device_id)?;

// NEW
let ctx = CudaContext::new(device_id)?;
let stream = ctx.default_stream();
```

3. **PTX Loading**:
```rust
// OLD
device.load_ptx(ptx_data, "module_name", &["kernel_name"])?;

// NEW
let module = CudaModule::from_ptx(&ctx, ptx_data, &[])?;
let kernel = module.get_function("kernel_name")?;
```

4. **Memory Allocation**:
```rust
// OLD
let d_buffer: CudaSlice<f32> = device.alloc_zeros(size)?;

// NEW
let d_buffer = ctx.alloc_zeros::<f32>(size)?;
```

5. **Host-to-Device Copy**:
```rust
// OLD
device.htod_sync_copy_into(&h_data, &mut d_buffer)?;

// NEW
stream.memcpy_htod(&d_buffer, &h_data)?;
stream.synchronize()?; // Only if sync needed
```

6. **Device-to-Host Copy**:
```rust
// OLD
device.dtoh_sync_copy(&d_buffer)?;

// NEW
let mut h_buffer = vec![0.0f32; size];
stream.memcpy_dtoh(&mut h_buffer, &d_buffer)?;
stream.synchronize()?;
```

7. **Kernel Launch**:
```rust
// OLD
unsafe {
    func.launch(config, (param1, param2, param3))?;
}

// NEW
unsafe {
    stream.launch(
        &func,
        config,
        (param1, param2, param3),
    )?;
}
```

### B.2.3 `crates/prism-gpu/src/whcr.rs`

**AGENT**: prism-gpu-specialist

**Full Transformation Template**:

```rust
//! WHCR GPU Module - cudarc 0.18+ Version
//!
//! Migrated from cudarc 0.9 synchronous API to 0.18+ stream-based async API.

use anyhow::Result;
use cudarc::driver::{
    CudaContext, CudaStream, CudaModule, CudaFunction,
    CudaView, CudaViewMut, LaunchConfig,
};
use std::sync::Arc;

/// WHCR execution context with stream support
pub struct WhcrContext {
    ctx: Arc<CudaContext>,
    stream: CudaStream,
    module: CudaModule,

    // Kernels
    count_conflicts_f32: CudaFunction,
    count_conflicts_f64: CudaFunction,
    evaluate_moves_f32: CudaFunction,
    evaluate_moves_f64: CudaFunction,
    apply_moves: CudaFunction,
    compute_wavelet_details: CudaFunction,
    compute_wavelet_priorities: CudaFunction,
}

impl WhcrContext {
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        let ptx_data = std::fs::read(ptx_path)?;
        let module = CudaModule::from_ptx(&ctx, &ptx_data, &[])?;

        // Create dedicated stream for WHCR operations
        let stream = ctx.new_stream()?;

        Ok(Self {
            ctx,
            stream,
            module,
            count_conflicts_f32: module.get_function("count_conflicts_f32")?,
            count_conflicts_f64: module.get_function("count_conflicts_f64")?,
            evaluate_moves_f32: module.get_function("evaluate_moves_f32")?,
            evaluate_moves_f64: module.get_function("evaluate_moves_f64")?,
            apply_moves: module.get_function("apply_moves_with_locking")?,
            compute_wavelet_details: module.get_function("compute_wavelet_details")?,
            compute_wavelet_priorities: module.get_function("compute_wavelet_priorities")?,
        })
    }

    /// Async conflict counting - returns immediately, use stream.synchronize() to wait
    pub fn count_conflicts_async(
        &self,
        coloring: &CudaView<i32>,
        row_ptr: &CudaView<i32>,
        col_idx: &CudaView<i32>,
        conflict_counts: &mut CudaViewMut<f32>,
        num_vertices: i32,
    ) -> Result<()> {
        let config = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((num_vertices as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch(
                &self.count_conflicts_f32,
                config,
                (
                    coloring.device_ptr(),
                    row_ptr.device_ptr(),
                    col_idx.device_ptr(),
                    conflict_counts.device_ptr_mut(),
                    num_vertices,
                ),
            )?;
        }

        Ok(())
    }

    /// Synchronize stream - wait for all async operations to complete
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    /// Get stream reference for external coordination
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}
```

### B.2.4 `crates/prism-gpu/src/thermodynamic.rs`

**AGENT**: prism-gpu-specialist

**Key Changes**:

```rust
// Stream-based parallel tempering
pub struct ThermodynamicContext {
    ctx: Arc<CudaContext>,
    streams: Vec<CudaStream>,  // One stream per replica
    module: CudaModule,
    // ... kernels
}

impl ThermodynamicContext {
    pub fn new(ctx: Arc<CudaContext>, num_replicas: usize, ptx_path: &str) -> Result<Self> {
        let ptx_data = std::fs::read(ptx_path)?;
        let module = CudaModule::from_ptx(&ctx, &ptx_data, &[])?;

        // Create one stream per replica for true parallel execution
        let streams: Vec<CudaStream> = (0..num_replicas)
            .map(|_| ctx.new_stream())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            ctx,
            streams,
            module,
            // ...
        })
    }

    /// Run parallel tempering step across all replicas concurrently
    pub fn parallel_tempering_step_async(&self, /* params */) -> Result<()> {
        for (replica_id, stream) in self.streams.iter().enumerate() {
            // Each replica runs on its own stream - TRUE parallelism
            unsafe {
                stream.launch(
                    &self.tempering_kernel,
                    config,
                    (/* params for this replica */),
                )?;
            }
        }
        // Returns immediately - all replicas running in parallel
        Ok(())
    }

    /// Wait for all replicas to complete
    pub fn synchronize_all(&self) -> Result<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }
}
```

### B.2.5 `crates/prism-gpu/src/quantum.rs`

**AGENT**: prism-gpu-specialist

**Migration Pattern** (same as above, apply stream-based pattern)

### B.2.6 `crates/prism-gpu/src/dendritic_reservoir.rs`

**AGENT**: prism-gpu-specialist

**Migration Pattern** (same as above, apply stream-based pattern)

### B.2.7 `foundation/quantum/src/lib.rs`

**AGENT**: prism-gpu-specialist

**Update cudarc imports and API calls following the patterns above**

### B.2.8 `foundation/neuromorphic/src/lib.rs`

**AGENT**: prism-gpu-specialist

**Update cudarc imports and API calls following the patterns above**

## B.3 Struct-Based Parameter Passing

### B.3.1 RuntimeConfig Definition

**File**: `crates/prism-core/src/runtime_config.rs`

**AGENT**: prism-architect

```rust
//! RuntimeConfig - GPU-transferable configuration struct
//!
//! This struct is designed to be copied to GPU constant memory or passed
//! as a single kernel parameter, eliminating the 12-parameter limit.

use serde::{Deserialize, Serialize};

/// RuntimeConfig for GPU kernel - must be #[repr(C)] for FFI compatibility
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RuntimeConfig {
    // ═══════════════════════════════════════════════════════════════════
    // WHCR Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub stress_weight: f32,           // Weight for geodesic stress in move evaluation
    pub persistence_weight: f32,      // Weight for TDA persistence scores
    pub belief_weight: f32,           // Weight for active inference beliefs
    pub hotspot_multiplier: f32,      // Bonus multiplier for hotspot vertices

    // ═══════════════════════════════════════════════════════════════════
    // Dendritic Reservoir Parameters (8-branch)
    // ═══════════════════════════════════════════════════════════════════
    pub tau_decay: [f32; 8],          // Time constants per branch
    pub branch_weights: [f32; 8],     // Soma integration weights
    pub reservoir_leak_rate: f32,     // Echo state leak rate
    pub spectral_radius: f32,         // Reservoir spectral radius
    pub input_scaling: f32,           // Input signal scaling
    pub reservoir_sparsity: f32,      // Recurrent connection sparsity

    // ═══════════════════════════════════════════════════════════════════
    // W-Cycle Multigrid Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub num_levels: i32,              // Number of multigrid levels (1-4)
    pub coarsening_ratio: f32,        // Vertex reduction per level
    pub restriction_weight: f32,      // Fine→coarse weight
    pub prolongation_weight: f32,     // Coarse→fine weight
    pub pre_smooth_iterations: i32,   // Smoothing before restriction
    pub post_smooth_iterations: i32,  // Smoothing after prolongation

    // ═══════════════════════════════════════════════════════════════════
    // Quantum Tunneling Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub tunneling_prob_base: f32,     // Base tunneling probability
    pub tunneling_prob_boost: f32,    // Boost at phase transitions
    pub chemical_potential: f32,      // Color index penalty (μ)
    pub transverse_field: f32,        // Quantum mixing strength
    pub interference_decay: f32,      // Decoherence rate
    pub num_quantum_states: i32,      // Superposition states per vertex

    // ═══════════════════════════════════════════════════════════════════
    // Parallel Tempering Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub temperatures: [f32; 16],      // Temperature ladder
    pub num_replicas: i32,            // Active replicas (1-16)
    pub swap_interval: i32,           // Iterations between swap attempts
    pub swap_probability: f32,        // Base swap acceptance

    // ═══════════════════════════════════════════════════════════════════
    // TPTP (Topological Phase Transition Prediction) Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub betti_0_threshold: f32,       // Connected components threshold
    pub betti_1_threshold: f32,       // Cycle threshold
    pub betti_2_threshold: f32,       // Void threshold
    pub persistence_threshold: f32,   // Min persistence for features
    pub stability_window: i32,        // Iterations for stability check
    pub transition_sensitivity: f32,  // Phase boundary detection

    // ═══════════════════════════════════════════════════════════════════
    // Active Inference Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub free_energy_threshold: f32,   // Expected free energy trigger
    pub belief_update_rate: f32,      // Belief distribution learning rate
    pub precision_weight: f32,        // Sensory precision weighting
    pub policy_temperature: f32,      // Action selection temperature

    // ═══════════════════════════════════════════════════════════════════
    // Meta/Control Parameters
    // ═══════════════════════════════════════════════════════════════════
    pub iteration: i32,               // Current iteration number
    pub phase_id: i32,                // Current phase (0-7)
    pub global_temperature: f32,      // Global annealing temperature
    pub learning_rate: f32,           // Online learning rate
    pub exploration_rate: f32,        // Epsilon for exploration

    // ═══════════════════════════════════════════════════════════════════
    // Flags (packed as i32 for GPU compatibility)
    // ═══════════════════════════════════════════════════════════════════
    pub flags: i32,                   // Bit flags for boolean options
    // Bit 0: enable_quantum_tunneling
    // Bit 1: enable_tptp
    // Bit 2: enable_dendritic_reservoir
    // Bit 3: enable_active_inference
    // Bit 4: enable_parallel_tempering
    // Bit 5: enable_multigrid
    // Bit 6: use_f64_precision
    // Bit 7: enable_online_learning

    // Padding to align to 256 bytes (cache line friendly)
    pub _padding: [f32; 4],
}

impl RuntimeConfig {
    /// Create default production configuration
    pub fn production() -> Self {
        Self {
            // WHCR
            stress_weight: 0.25,
            persistence_weight: 0.1,
            belief_weight: 0.3,
            hotspot_multiplier: 1.2,

            // Dendritic (8 branches with varying time constants)
            tau_decay: [0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98],
            branch_weights: [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            reservoir_leak_rate: 0.3,
            spectral_radius: 0.9,
            input_scaling: 0.3,
            reservoir_sparsity: 0.1,

            // W-Cycle
            num_levels: 4,
            coarsening_ratio: 0.25,
            restriction_weight: 0.5,
            prolongation_weight: 0.5,
            pre_smooth_iterations: 3,
            post_smooth_iterations: 3,

            // Quantum
            tunneling_prob_base: 0.05,
            tunneling_prob_boost: 2.0,
            chemical_potential: 2.5,
            transverse_field: 0.1,
            interference_decay: 0.01,
            num_quantum_states: 6,

            // Parallel Tempering
            temperatures: [
                0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0,
                1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0,
            ],
            num_replicas: 12,
            swap_interval: 10,
            swap_probability: 0.5,

            // TPTP
            betti_0_threshold: 1.0,
            betti_1_threshold: 0.1,
            betti_2_threshold: 0.01,
            persistence_threshold: 0.05,
            stability_window: 20,
            transition_sensitivity: 0.8,

            // Active Inference
            free_energy_threshold: 0.5,
            belief_update_rate: 0.1,
            precision_weight: 1.0,
            policy_temperature: 1.0,

            // Meta
            iteration: 0,
            phase_id: 0,
            global_temperature: 1.0,
            learning_rate: 0.01,
            exploration_rate: 0.2,

            // Flags: all features enabled
            flags: 0b11111111,

            _padding: [0.0; 4],
        }
    }

    // Flag accessors
    pub fn quantum_enabled(&self) -> bool { self.flags & (1 << 0) != 0 }
    pub fn tptp_enabled(&self) -> bool { self.flags & (1 << 1) != 0 }
    pub fn dendritic_enabled(&self) -> bool { self.flags & (1 << 2) != 0 }
    pub fn active_inference_enabled(&self) -> bool { self.flags & (1 << 3) != 0 }
    pub fn tempering_enabled(&self) -> bool { self.flags & (1 << 4) != 0 }
    pub fn multigrid_enabled(&self) -> bool { self.flags & (1 << 5) != 0 }
    pub fn f64_precision(&self) -> bool { self.flags & (1 << 6) != 0 }
    pub fn online_learning(&self) -> bool { self.flags & (1 << 7) != 0 }
}

/// Telemetry output from kernel - also GPU-transferable
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KernelTelemetry {
    pub conflicts: i32,
    pub colors_used: i32,
    pub moves_applied: i32,
    pub tunneling_events: i32,
    pub phase_transitions: i32,
    pub betti_numbers: [f32; 3],
    pub reservoir_activity: f32,
    pub free_energy: f32,
    pub best_replica: i32,
    pub iteration_time_us: i32,
    pub _padding: [f32; 2],
}
```

### B.3.2 CUDA Header for RuntimeConfig

**File**: `crates/prism-gpu/src/kernels/runtime_config.cuh`

**AGENT**: prism-gpu-specialist

```cuda
#ifndef RUNTIME_CONFIG_CUH
#define RUNTIME_CONFIG_CUH

/**
 * RuntimeConfig - GPU-side mirror of Rust RuntimeConfig
 * MUST be kept in sync with crates/prism-core/src/runtime_config.rs
 */
struct RuntimeConfig {
    // WHCR Parameters
    float stress_weight;
    float persistence_weight;
    float belief_weight;
    float hotspot_multiplier;

    // Dendritic Reservoir (8-branch)
    float tau_decay[8];
    float branch_weights[8];
    float reservoir_leak_rate;
    float spectral_radius;
    float input_scaling;
    float reservoir_sparsity;

    // W-Cycle Multigrid
    int num_levels;
    float coarsening_ratio;
    float restriction_weight;
    float prolongation_weight;
    int pre_smooth_iterations;
    int post_smooth_iterations;

    // Quantum Tunneling
    float tunneling_prob_base;
    float tunneling_prob_boost;
    float chemical_potential;
    float transverse_field;
    float interference_decay;
    int num_quantum_states;

    // Parallel Tempering
    float temperatures[16];
    int num_replicas;
    int swap_interval;
    float swap_probability;

    // TPTP
    float betti_0_threshold;
    float betti_1_threshold;
    float betti_2_threshold;
    float persistence_threshold;
    int stability_window;
    float transition_sensitivity;

    // Active Inference
    float free_energy_threshold;
    float belief_update_rate;
    float precision_weight;
    float policy_temperature;

    // Meta/Control
    int iteration;
    int phase_id;
    float global_temperature;
    float learning_rate;
    float exploration_rate;

    // Flags
    int flags;

    // Padding
    float _padding[4];
};

// Flag bit accessors
__device__ __forceinline__ bool quantum_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 0)) != 0;
}

__device__ __forceinline__ bool tptp_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 1)) != 0;
}

__device__ __forceinline__ bool dendritic_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 2)) != 0;
}

__device__ __forceinline__ bool active_inference_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 3)) != 0;
}

__device__ __forceinline__ bool tempering_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 4)) != 0;
}

__device__ __forceinline__ bool multigrid_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 5)) != 0;
}

/**
 * Kernel Telemetry - output from GPU
 */
struct KernelTelemetry {
    int conflicts;
    int colors_used;
    int moves_applied;
    int tunneling_events;
    int phase_transitions;
    float betti_numbers[3];
    float reservoir_activity;
    float free_energy;
    int best_replica;
    int iteration_time_us;
    float _padding[2];
};

#endif // RUNTIME_CONFIG_CUH
```

## B.4 Verification Checklist

**AGENT**: All agents must verify these after Phase 0 completion

- [ ] `cargo check` passes with no errors
- [ ] All PTX files compile with nvcc
- [ ] Basic kernel launch test passes (count_conflicts)
- [ ] Stream synchronization works correctly
- [ ] RuntimeConfig struct transfers to GPU correctly
- [ ] Telemetry struct returns from GPU correctly
- [ ] Multi-stream parallel execution works
- [ ] Memory allocation/deallocation doesn't leak

## B.5 Migration Commands

```bash
# Step 1: Update Cargo.toml dependencies
cd /mnt/c/Users/Predator/Desktop/PRISM

# Step 2: Run cargo check to see all errors
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
cargo check 2>&1 | tee migration_errors.log

# Step 3: Count errors by file
grep "^error" migration_errors.log | wc -l

# Step 4: After fixing, rebuild PTX
for kernel in whcr quantum thermodynamic dendritic_whcr tda; do
    /usr/local/cuda-12.6/bin/nvcc --ptx \
        -o target/ptx/$kernel.ptx \
        crates/prism-gpu/src/kernels/$kernel.cu \
        -arch=sm_86 --std=c++14 -Xcompiler -fPIC
done

# Step 5: Full build
PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH" \
CUDA_HOME=/usr/local/cuda-12.6 \
cargo build --release
```

---

# END OF PART 1

**Next**: Part 2 covers Phase 1 (Ultra Kernel Components)
