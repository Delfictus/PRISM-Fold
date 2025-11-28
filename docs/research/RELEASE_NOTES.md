# PRISM-LBS Release Notes

## v0.1.0 — Initial GPU-Integrated Release

### Highlights
- CUDA kernels for LBS geometry: surface accessibility, distance matrix, pocket clustering, and druggability scoring (`prism-gpu/src/kernels/lbs/*`, PTX via `compile_ptx.sh`).
- Host bindings: `prism-gpu/src/lbs.rs` loads PTX modules and exposes `LbsGpu` operations for surfaces, distances, clustering, and scoring.
- GPU-aware predictor: `prism-lbs/src/lib.rs` now uses `SurfaceComputer::compute_gpu` and GPU distance matrices when a `GpuContext` is available; CPU fallback remains intact.
- Graph builder GPU path: `ProteinGraphBuilder::build_with_gpu` consumes shared GPU context when `graph.use_gpu` is enabled.
- Pipeline integration: `run_lbs_with_gpu` lets the orchestrator reuse its GPU context for LBS runs.
- CLI toggles: `--gpu-geometry`, `--cpu-geometry`, `--gpu-device`, `--ptx-dir`, and `--cpu` control GPU usage; config adds `graph.use_gpu` alongside `use_gpu`.
- Validation: Added CUDA smoke test (`tests/gpu_path.rs`) plus benchmark harness (`prism-lbs-benchmark`).

### Upgrading / Usage Notes
- Build PTX before GPU runs: `./compile_ptx.sh` (outputs to `target/ptx`).
- Set GPU discovery via env: `PRISM_PTX_DIR`, `PRISM_GPU_DEVICE` (or CLI flags `--ptx-dir`, `--gpu-device`).
- Enable GPU geometry/graph from CLI with `--gpu-geometry`; disable with `--cpu-geometry` or `--cpu`. Config toggle: `graph.use_gpu = true`.
- CPU fallback is automatic if GPU init or kernels are unavailable; a warning is logged.

### Known Issues / Caveats
- Workspace emits warnings in non-LBS crates; they do not affect LBS functionality.
- GPU tests require CUDA hardware and PTX artifacts present in `PRISM_PTX_DIR`.
- PTX path defaults to `target/ptx`; update if deploying in containerized environments.
