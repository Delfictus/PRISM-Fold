# Multi-GPU Integration Guide

## Overview

PRISM now supports multi-GPU execution through the `MultiGpuContext` integration layer. This enables:

- **Automatic GPU Detection**: Discovers all available CUDA devices
- **Graceful Fallback**: Degrades to single-GPU mode when needed
- **Work Distribution**: Load-balanced distribution across GPUs
- **Replica Exchange**: Coordinated parallel tempering with P2P transfers
- **Zero-Code Migration**: Existing single-GPU code works without changes

## Quick Start

### Auto-Detection Mode

```rust
use prism_gpu::MultiGpuContext;

// Auto-detect and initialize all available GPUs
let mut ctx = MultiGpuContext::new_auto()?;

println!("Running on {} GPU(s)", ctx.num_devices());

// Get primary device (for single-GPU operations)
let device = ctx.primary_device();
```

### Explicit Multi-GPU Mode

```rust
// Use specific GPUs with 16 replicas for parallel tempering
let mut ctx = MultiGpuContext::new(&[0, 1, 2], 16)?;

// Check if multi-GPU mode is active
if ctx.is_multi_gpu() {
    println!("Multi-GPU enabled with {} devices", ctx.num_devices());
}
```

## Work Distribution

### Round-Robin Distribution

```rust
let graphs = vec![graph1, graph2, graph3, graph4, graph5, graph6];
let distribution = ctx.distribute(&graphs);

for (device_id, graphs_for_device) in distribution {
    println!("GPU {}: {} graphs", device_id, graphs_for_device.len());
    // Launch kernels on this device...
}
```

### Weighted Distribution

```rust
// GPU 0 gets 2x capacity of other GPUs
let distribution = ctx.distribute_weighted(&graphs, |device_id| {
    if device_id == 0 { 2.0 } else { 1.0 }
});
```

## Parallel Tempering Integration

### Thermodynamic Phase (Phase 2)

```rust
use prism_gpu::{MultiGpuContext, ThermodynamicGpu};

// Initialize multi-GPU context
let mut ctx = MultiGpuContext::new_auto()?;

// Create thermodynamic GPU instance per device
let mut thermo_gpus: Vec<ThermodynamicGpu> = Vec::new();
for i in 0..ctx.num_devices() {
    let device = ctx.device(i);
    let thermo = ThermodynamicGpu::new(device, "kernels/ptx/thermodynamic.ptx")?;
    thermo_gpus.push(thermo);
}

// Run parallel tempering across GPUs
for iteration in 0..10000 {
    // Step 1: Execute annealing on each GPU (parallel)
    for (device_id, thermo) in thermo_gpus.iter_mut().enumerate() {
        let replicas = ctx.coordinator()
            .unwrap()
            .device_replicas(device_id);

        for &replica in replicas {
            // Run thermodynamic step on this replica
            thermo.parallel_tempering_step(replica)?;
        }
    }

    // Step 2: Exchange replicas across GPUs
    ctx.parallel_tempering_step()?;

    // Step 3: Synchronize all GPUs
    ctx.synchronize_all()?;
}
```

### Replica Mapping

```rust
// Check which GPU hosts each replica
if let Some(mapping) = ctx.replica_mapping() {
    for (replica, &device) in mapping.iter().enumerate() {
        println!("Replica {} on GPU {}", replica, device);
    }
}

// Get device hosting specific replica
let device_id = ctx.replica_device(5); // Replica 5
```

## CLI Integration

### Command Line Options

```bash
# Auto-detect all GPUs
prism-cli --input graph.col --gpu

# Use specific GPUs
prism-cli --input graph.col --gpu --gpu-devices 0,1,2

# Single GPU fallback
prism-cli --input graph.col --gpu --gpu-devices 0

# With scheduling policy
prism-cli --input graph.col --gpu --gpu-devices 0,1,2 --gpu-scheduling-policy least-loaded
```

### Configuration Example

```rust
// In main.rs CLI integration
let device_ids = args.gpu_devices; // From --gpu-devices 0,1,2
let num_replicas = args.phase2_replicas; // From --phase2-replicas 12

let mut gpu_ctx = if device_ids.len() > 1 {
    // Multi-GPU mode
    MultiGpuContext::new(&device_ids, num_replicas)?
} else {
    // Single-GPU mode (auto-detect or explicit)
    MultiGpuContext::new_auto()?
};

// Pass to pipeline orchestrator
orchestrator.set_gpu_context(gpu_ctx);
```

## P2P Capabilities

### Checking P2P Status

```rust
if let Some(pool) = ctx.pool() {
    for i in 0..ctx.num_devices() {
        for j in (i+1)..ctx.num_devices() {
            if pool.can_p2p(i, j) {
                println!("P2P enabled: GPU {} ↔ GPU {} ({:.1} GB/s)",
                    i, j, pool.p2p_bandwidth(i, j));
            } else {
                println!("P2P disabled: GPU {} ↔ GPU {} (using CPU staging)", i, j);
            }
        }
    }
}
```

### P2P Replica Exchange

Replica exchange automatically uses P2P when available:

```rust
// This call uses P2P if available, falls back to CPU staging otherwise
ctx.parallel_tempering_step()?;
```

## Performance Considerations

### GPU Utilization

- **Multi-GPU Scaling**: Near-linear scaling for embarrassingly parallel problems
- **P2P Overhead**: ~5-10% overhead for cross-GPU replica exchanges
- **CPU Staging Fallback**: ~20-30% overhead when P2P unavailable

### Memory Requirements

- Each GPU needs enough VRAM for its assigned replicas
- Typical: 512 MB per replica for DSJC500-class graphs
- Example: 4 GPUs × 3 replicas × 512 MB = 6 GB total VRAM

### Best Practices

1. **Use P2P-capable GPUs**: NVLink or PCIe 4.0 x16 for best performance
2. **Balance replica count**: Target 3-4 replicas per GPU
3. **Synchronize strategically**: Minimize `synchronize_all()` calls
4. **Monitor GPU utilization**: Use `nvidia-smi` to ensure all GPUs are active

## Migration Path

### From Single-GPU

**Before:**
```rust
let device = CudaDevice::new(0)?;
let thermo = ThermodynamicGpu::new(device, "kernels/ptx/thermodynamic.ptx")?;
```

**After:**
```rust
let ctx = MultiGpuContext::new_auto()?;
let device = ctx.primary_device();
let thermo = ThermodynamicGpu::new(device, "kernels/ptx/thermodynamic.ptx")?;
// Works exactly the same in single-GPU mode!
```

### From Explicit Device Selection

**Before:**
```rust
let device = CudaDevice::new(args.gpu_device)?;
```

**After:**
```rust
let ctx = MultiGpuContext::new(&[args.gpu_device], 12)?;
let device = ctx.primary_device();
```

## Advanced Usage

### Dynamic Replica Redistribution

```rust
// Reassign replica 5 to GPU 2 (for load balancing)
if let Some(coord) = ctx.coordinator_mut() {
    coord.reassign_replica(5, 2)?;
}
```

### Custom Number of Replicas

```rust
// Start with 8 replicas
let mut ctx = MultiGpuContext::new(&[0, 1], 8)?;

// Later, increase to 16 replicas
ctx.set_num_replicas(16)?;
```

### Accessing Individual Devices

```rust
// Get all devices for manual kernel management
let devices = ctx.devices();
for (i, device) in devices.iter().enumerate() {
    println!("Device {}: {:?}", i, device);
}

// Get specific device
let device_2 = ctx.device(2);
```

## Troubleshooting

### Issue: "Failed to initialize multi-GPU pool"

**Cause**: GPU detection or initialization failed
**Solution**: Check CUDA drivers, verify `nvidia-smi` shows all GPUs
**Fallback**: System automatically falls back to single-GPU mode

### Issue: "P2P disabled" warnings

**Cause**: GPUs not P2P-capable or on different PCIe root complexes
**Impact**: Replica exchanges use slower CPU staging
**Mitigation**: Use GPUs with NVLink or ensure same PCIe topology

### Issue: Uneven performance across GPUs

**Cause**: Imbalanced replica distribution or thermal throttling
**Solution**: Use `distribute_weighted()` with custom capacity weights
**Monitoring**: Check GPU utilization with `nvidia-smi dmon`

## Testing

### Unit Tests

```bash
# Run multi-GPU integration tests (requires GPUs)
cargo test -p prism-gpu multi_gpu_integration --ignored

# Run logic tests (no GPU required)
cargo test -p prism-gpu multi_gpu_integration
```

### Benchmarks

```bash
# Compare single-GPU vs multi-GPU performance
cargo run --release --example multi_gpu_benchmark -- \
    --input benchmarks/DSJC500.5.col \
    --single-gpu \
    --multi-gpu 0,1,2
```

## Examples

See `examples/multi_gpu_demo.rs` for complete working example:

```bash
cargo run --release --example multi_gpu_demo
```

## Reference

- **Module**: `crates/prism-gpu/src/multi_gpu_integration.rs`
- **Dependencies**: `multi_device_pool.rs`, `stream_manager.rs`
- **CLI Integration**: `crates/prism-cli/src/main.rs` lines 243-262
- **Pipeline Integration**: To be implemented in Phase 1B

---

**Last Updated**: 2025-11-29
**PRISM Version**: 2.0
**Author**: PRISM Integration Team
