# World Record Pipeline Configurations

This directory contains configuration files for the PRISM World Record Breaking Pipeline.

## Available Configurations

### `world_record.v1.toml` / `world_record.v1.json`
**Full world-record attempt configuration**
- Target: 83 colors (DSJC1000.5 world record)
- Runtime: 48 hours maximum
- Workers: 24 threads
- All phases enabled (GPU Reservoir, Active Inference, ADP, Thermodynamic, Quantum-Classical, Multi-Scale, Ensemble)

### `quick_test.toml`
**Quick validation configuration**
- Target: 100 colors (relaxed for faster testing)
- Runtime: 1 hour maximum
- Workers: 8 threads
- Minimal phases enabled (Reservoir + Quantum-Classical only)

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

## Configuration Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `target_chromatic` | usize | ≥ 1 | Target chromatic number (world record goal) |
| `max_runtime_hours` | f64 | (0, 168] | Maximum runtime in hours (up to 1 week) |
| `num_workers` | usize | [1, 256] | Number of parallel worker threads |
| `use_reservoir_prediction` | bool | - | Enable GPU-accelerated neuromorphic reservoir |
| `use_active_inference` | bool | - | Enable Active Inference policy selection |
| `use_adp_learning` | bool | - | Enable ADP Q-learning parameter tuning |
| `use_thermodynamic_equilibration` | bool | - | Enable thermodynamic replica exchange |
| `use_quantum_classical_hybrid` | bool | - | Enable quantum-classical feedback loop |
| `use_multiscale_analysis` | bool | - | Enable multi-scale neuromorphic analysis |
| `use_ensemble_consensus` | bool | - | Enable ensemble consensus voting |

## Validation

All configurations are validated on load:
- `target_chromatic` must be ≥ 1
- `max_runtime_hours` must be in (0, 168]
- `num_workers` must be in [1, 256]
- At least one phase must be enabled

Invalid configurations will fail with a descriptive error message.

## Creating Custom Configurations

1. Copy an existing config file
2. Modify parameters as needed
3. Run with your custom config path
4. The loader auto-detects TOML/JSON based on extension

Example TOML:
```toml
target_chromatic = 90
max_runtime_hours = 24.0
num_workers = 16
use_reservoir_prediction = true
use_active_inference = true
# ... other flags
```

Example JSON:
```json
{
  "target_chromatic": 90,
  "max_runtime_hours": 24.0,
  "num_workers": 16,
  "use_reservoir_prediction": true
}
```
