# World Record Pipeline Configurations

This directory contains configuration files for the PRISM World Record Breaking Pipeline with comprehensive nested configuration support.

## Available Configurations

### `world_record.v1.toml` / `world_record.v1.json`
**Full world-record attempt configuration**
- Profile: "record"
- Target: 83 colors (DSJC1000.5 world record)
- Runtime: 48 hours maximum
- Workers: 24 threads
- All phases enabled (GPU Reservoir, Active Inference, ADP, Thermodynamic, Quantum-Classical, Multi-Scale, Ensemble)
- Optimized parameters for maximum performance

### `quick_test.toml`
**Quick validation configuration**
- Profile: "quick_test"
- Target: 100 colors (relaxed for faster testing)
- Runtime: 1 hour maximum
- Workers: 8 threads
- Minimal phases enabled (Reservoir + Quantum-Classical only)
- Reduced parameters for fast iteration

## Usage

### Default (uses world_record.v1.toml)
```bash
cargo run --release --features cuda -p prct-core --example world_record_dsjc1000
```

### Specify custom config
```bash
cargo run --release --features cuda -p prct-core --example world_record_dsjc1000 configs/quick_test.toml
```

### Use JSON format
```bash
cargo run --release --features cuda -p prct-core --example world_record_dsjc1000 configs/world_record.v1.json
```

## Configuration Structure

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `profile` | string | "record" | Configuration profile name |
| `version` | string | "1.0.0" | Configuration version |
| `target_chromatic` | usize | 83 | Target chromatic number (world record goal) |
| `deterministic` | bool | false | Use deterministic mode with fixed seed |
| `seed` | u64 | 123456789 | Random seed for deterministic mode |
| `max_runtime_hours` | f64 | 48.0 | Maximum runtime in hours (up to 1 week) |
| `num_workers` | usize | 24 | Number of parallel worker threads |
| `use_reservoir_prediction` | bool | true | Enable GPU-accelerated neuromorphic reservoir |
| `use_active_inference` | bool | true | Enable Active Inference policy selection |
| `use_adp_learning` | bool | true | Enable ADP Q-learning parameter tuning |
| `use_thermodynamic_equilibration` | bool | true | Enable thermodynamic replica exchange |
| `use_quantum_classical_hybrid` | bool | true | Enable quantum-classical feedback loop |
| `use_multiscale_analysis` | bool | true | Enable multi-scale neuromorphic analysis |
| `use_ensemble_consensus` | bool | true | Enable ensemble consensus voting |

### `[gpu]` - GPU Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_id` | usize | 0 | CUDA device ID |
| `streams` | usize | 3 | Number of CUDA streams for parallelism |
| `batch_size` | usize | 1024 | Batch size for GPU operations |
| `enable_reservoir_gpu` | bool | true | Enable GPU acceleration for reservoir |
| `enable_thermo_gpu` | bool | true | Enable GPU acceleration for thermodynamic |
| `enable_quantum_gpu` | bool | true | Enable GPU acceleration for quantum |

### `[memetic]` - Memetic Algorithm Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `population_size` | usize | 128 | Population size (32-64 recommended) |
| `elite_size` | usize | 16 | Elite size (top 10-20%) |
| `generations` | usize | 500 | Number of generations |
| `mutation_rate` | f64 | 0.20 | Mutation rate (0.1-0.3) |
| `tournament_size` | usize | 5 | Tournament size for selection |
| `local_search_depth` | usize | 50000 | DSATUR iterations per generation |
| `use_tsp_guidance` | bool | true | Enable TSP-guided operators |
| `tsp_weight` | f64 | 0.25 | Weight for TSP quality in fitness (0.0-1.0) |

### `[thermo]` - Thermodynamic Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `replicas` | usize | 64 | Number of parallel replicas |
| `num_temps` | usize | 64 | Number of temperature levels |
| `exchange_interval` | usize | 50 | Steps between replica exchanges |
| `t_min` | f64 | 0.001 | Minimum temperature |
| `t_max` | f64 | 1.0 | Maximum temperature |

### `[quantum]` - Quantum Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iterations` | usize | 20 | Quantum annealing iterations |
| `target_chromatic` | usize | 83 | Target chromatic for QUBO formulation |

### `[adp]` - ADP Q-Learning Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epsilon` | f64 | 1.0 | Initial exploration rate |
| `epsilon_decay` | f64 | 0.995 | Exploration decay factor |
| `epsilon_min` | f64 | 0.03 | Minimum exploration rate |
| `alpha` | f64 | 0.10 | Learning rate |
| `gamma` | f64 | 0.95 | Discount factor |

### `[orchestrator]` - Orchestrator Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adp_dsatur_depth` | usize | 200000 | DSATUR search depth for ADP |
| `adp_quantum_iterations` | usize | 20 | Quantum iterations for ADP |
| `adp_thermo_num_temps` | usize | 64 | Temperature levels for ADP |
| `restarts` | usize | 10 | Number of pipeline restarts |
| `early_stop_no_improve_iters` | usize | 3 | Early stop after N non-improving iterations |
| `checkpoint_minutes` | usize | 15 | Checkpoint interval in minutes |

## Validation

All configurations are validated on load:
- `target_chromatic` must be ≥ 1
- `max_runtime_hours` must be in (0, 168] hours (up to 1 week)
- `num_workers` must be in [1, 256]
- At least one phase must be enabled

Invalid configurations will fail with a descriptive error message.

## Creating Custom Configurations

1. Copy an existing config file (TOML or JSON)
2. Modify parameters as needed
3. Run with your custom config path
4. The loader auto-detects TOML/JSON based on extension

### Example: Custom Mid-Range Configuration

```toml
profile = "mid_range"
version = "1.0.0"
target_chromatic = 90
max_runtime_hours = 12.0
num_workers = 16

use_reservoir_prediction = true
use_active_inference = true
use_adp_learning = false
use_thermodynamic_equilibration = true
use_quantum_classical_hybrid = true
use_multiscale_analysis = false
use_ensemble_consensus = true

[gpu]
device_id = 0
streams = 2
batch_size = 768

[memetic]
population_size = 64
elite_size = 8
generations = 200
mutation_rate = 0.18
local_search_depth = 20000

[thermo]
replicas = 32
num_temps = 32
t_min = 0.005
t_max = 0.8

# ... other sections with adjusted parameters
```

## Configuration Profiles

### Record (world_record.v1.toml)
- **Goal**: Beat world record (83 colors)
- **Strategy**: Maximum resources, all phases enabled
- **Runtime**: 48 hours
- **Use Case**: Production world record attempts

### Quick Test (quick_test.toml)
- **Goal**: Fast validation (100 colors acceptable)
- **Strategy**: Minimal resources, core phases only
- **Runtime**: 1 hour
- **Use Case**: Development, testing, debugging

## Advanced Usage

### Deterministic Runs
Set `deterministic = true` and specify a `seed`:
```toml
deterministic = true
seed = 42
```

### GPU Multi-Stream Optimization
Increase streams for better GPU utilization:
```toml
[gpu]
streams = 4
batch_size = 2048
```

### Adaptive Parameters
The ADP section controls reinforcement learning:
```toml
[adp]
epsilon = 1.0          # Start with full exploration
epsilon_decay = 0.995  # Decay slowly
epsilon_min = 0.03     # Always keep 3% exploration
alpha = 0.10           # Conservative learning rate
gamma = 0.95           # Value future rewards highly
```

## Troubleshooting

### Config Parse Error
- Check TOML/JSON syntax
- Ensure all required fields are present
- Use the loader's error messages to locate issues

### Validation Failed
- Verify `target_chromatic >= 1`
- Ensure `max_runtime_hours` is in (0, 168]
- Check `num_workers` is in [1, 256]
- Enable at least one optimization phase

### Performance Issues
- Increase `num_workers` for CPU parallelism
- Adjust `[gpu].batch_size` for GPU efficiency
- Reduce `[memetic].population_size` for faster generations
- Decrease `[orchestrator].restarts` for shorter runs

---

**Last Updated**: 2025-11-01  
**Version**: 1.0.0
