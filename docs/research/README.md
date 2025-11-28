# PRISM-LBS: GPU-Accelerated Ligand Binding Site Prediction

Production-ready ligand binding site (LBS) platform built on PRISM’s GPU/quantum/neuro stack. CUDA kernels power surface accessibility, distance matrices, pocket clustering, and druggability scoring; a multi-phase detector and scorer deliver ranked pockets with CPU fallback.

## Highlights
- **CUDA-first geometry**: Surface accessibility (Shrake-Rupley style), distance matrices, pocket clustering, and druggability scoring via `prism-gpu` LBS kernels (`target/ptx`).
- **End-to-end phases**: Surface reservoir → pocket belief sampling → cavity/TDA analysis → WHCR refinement → druggability scoring.
- **Configurable acceleration**: CLI flags and TOML (`use_gpu`, `graph.use_gpu`) plus env (`PRISM_PTX_DIR`, `PRISM_GPU_DEVICE`) to run GPU when available and fall back cleanly.
- **Pipeline integration**: `prism_lbs::pipeline_integration::{run_lbs, run_lbs_with_gpu}` hooks into the orchestrator (`lbs` feature).
- **Bench/validation**: `prism-lbs-benchmark` binary and CUDA smoke test for surface/graph GPU paths.

## Prerequisites
- Rust 1.70+
- CUDA Toolkit 12.0+ and an NVIDIA GPU (Compute Capability ≥ 7.0 recommended)
- PTX artifacts built via `./compile_ptx.sh` (emits to `target/ptx`)

## Build
```bash
# From repo root
./compile_ptx.sh                                   # build LBS PTX
cargo build -p prism-lbs --release --features cuda # GPU path
cargo build -p prism-lbs --release --no-default-features # CPU-only
```

## Run (CLI)
```bash
# CPU geometry/graph
prism-lbs --input data/protein.pdb --output results/output.pdb --cpu-geometry

# GPU geometry + GPU graph (defaults: device 0, PTX in target/ptx)
PRISM_PTX_DIR=target/ptx PRISM_GPU_DEVICE=0 \
prism-lbs --input data/protein.pdb --output results/output.pdb --gpu-geometry

# Custom config and formats
prism-lbs --input sample.pdb --output out/ \
  --config prism-lbs/configs/default.toml \
  --format pdb,json
```

Key flags/env:
- `--gpu-geometry` / `--cpu-geometry` control SASA/geometry path.
- `--cpu` disables all GPU use.
- `--gpu-device <id>` / `--ptx-dir <dir>` override device/PTX discovery (or use env `PRISM_GPU_DEVICE`, `PRISM_PTX_DIR`).
- Graph GPU toggle lives in config `graph.use_gpu` (auto-follows `use_gpu` unless explicitly set).

## Tests
```bash
# CPU tests
cargo test -p prism-lbs --tests

# GPU smoke (needs hardware + PTX)
PRISM_PTX_DIR=target/ptx PRISM_GPU_DEVICE=0 \
cargo test -p prism-lbs --tests --features cuda -- gpu_path
```

## Benchmarks
```bash
cargo run -p prism-lbs --bin prism-lbs-benchmark --release
```

## Repository Pointers (LBS-specific)
- LBS crate: `prism-lbs/` (pipeline, phases, scoring, CLI)
- GPU bindings: `prism-gpu/src/lbs.rs`
- LBS CUDA kernels: `prism-gpu/src/kernels/lbs/`
- Surface/graph GPU paths: `prism-lbs/src/structure/surface.rs`, `prism-lbs/src/graph/protein_graph.rs`
- CLI: `prism-lbs/src/bin/main.rs`
- Pipeline hook: `prism-lbs/src/pipeline_integration.rs`

Other directories come from the broader PRISM platform; the focus of this repo is the LBS system and its GPU integration. See `RELEASE_NOTES.md` for the current release summary.
