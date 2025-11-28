# Deep Coupling → Tuning Pipeline Integration Checklist

**Branch**: `fluxnet-v2-feature`
**Target**: Integration with tuning pipeline
**Date**: 2025-11-19

---

## 📦 Deliverables Overview

This branch provides the **Deep Metaphysical Telemetry Coupling** feature, which implements a reflexive feedback loop where geometric stress from Phase 4/6 influences all phases and shapes FluxNet RL rewards.

**Key Deliverables**:
1. ✅ Geometry coupling implementation (all phases)
2. ✅ FluxNet RL reward shaping with geometry bonuses
3. 🔄 Retrained Q-table with geometry awareness (IN PROGRESS)
4. ✅ Deep coupling configuration (dsjc250_deep_coupling.toml)
5. ✅ Baseline GPU results (41 colors, 16 attempts)
6. ✅ Integration documentation
7. ⏳ Q-table validation runs (PENDING)

---

## ✅ Completed Tasks

### 1. Feature Implementation
- [x] Core geometry telemetry types (prism-core/src/types.rs)
- [x] Phase context geometry metrics (prism-core/src/traits.rs)
- [x] UniversalRLState geometry fields (prism-fluxnet/src/core/state.rs)
- [x] Reward shaping in FluxNet controller (prism-fluxnet/src/core/controller.rs)
- [x] Early-phase geometry seeding (Phase 1)
- [x] Temperature adjustment (Phase 2)
- [x] Warmstart hotspot prioritization (Phase 0)
- [x] Memetic mutation boost (Phase 7)
- [x] Continuous geometry propagation (orchestrator)
- [x] MetaphysicalCouplingConfig (prism-pipeline/src/config/mod.rs)

### 2. GPU Verification
- [x] Build with CUDA features (`cargo build --release --features cuda`)
- [x] Compile all 5 PTX kernels (thermodynamic, quantum, floyd_warshall, tda, dendritic_reservoir)
- [x] Verify PTX module loading
- [x] Confirm real GPU execution (Phase 2: ~93s, not <1ms CPU fallback)
- [x] Document GPU timing results

### 3. Baseline Experiments
- [x] Run DSJC250.5 with deep coupling, no Q-table (16 attempts)
- [x] Record chromatic results: 41 colors, 0 conflicts
- [x] Verify geometry coupling active (temp adjustment 10.16 → 12.67)
- [x] Archive logs: `artifacts/logs/baseline_gpu_16attempts_noqtable.log`
- [x] Create results summary: `artifacts/fluxnet/baseline_results_summary.txt`

### 4. FluxNet RL Training
- [x] Start Q-table training (1000 epochs, DSJC250.5)
- [x] Configure geometry reward shaping (built into training binary)
- [x] Set up training log: `artifacts/logs/fluxnet_train_v3.log`
- [x] Output file: `artifacts/fluxnet/curriculum_bank_v3_geometry.bin`

### 5. Documentation
- [x] Create integration guide: `docs/deep_coupling_integration_notes.md`
- [x] Document training parameters: `artifacts/fluxnet/README.md`
- [x] Create integration checklist: `INTEGRATION_CHECKLIST.md` (this file)
- [x] Archive baseline results summary

---

## 🔄 In Progress

### 1. Q-Table Training
- **Status**: Running (epoch 3/1000 as of last check)
- **Expected Duration**: 2-4 hours
- **Command**: `./target/release/fluxnet_train benchmarks/dimacs/DSJC250.5.col 1000 artifacts/fluxnet/curriculum_bank_v3_geometry.bin`
- **Log**: `artifacts/logs/fluxnet_train_v3.log`
- **Monitor**: `tail -f artifacts/logs/fluxnet_train_v3.log`

---

## ⏳ Pending Tasks (Awaiting Q-Table Completion)

### 1. Q-Table Validation Runs
- [ ] Run DSJC250.5 with Q-table (16 attempts)
  - [ ] Record chromatic number
  - [ ] Compare vs baseline (41 colors)
  - [ ] Check for geometry reward bonus logs
  - [ ] Log file: `artifacts/logs/gpu_run_with_qtable_16att.log`

- [ ] Run DSJC250.5 with Q-table (128 attempts, memetic enabled)
  - [ ] Record best chromatic number
  - [ ] Measure convergence speed
  - [ ] Compare vs baseline extrapolated to 128 attempts
  - [ ] Log file: `artifacts/logs/gpu_run_with_qtable_128att.log`

- [ ] Extract telemetry/stress metrics
  - [ ] Investigate telemetry JSONL export
  - [ ] Use profiler if JSONL not working: `--enable-profiler --profiler-output artifacts/telemetry/profile_qtable.json`

### 2. Results Analysis
- [ ] Document chromatic improvements (Q-table vs baseline)
- [ ] Analyze geometry stress trajectories
- [ ] Check FluxNet RL reward logs for geometry bonuses
- [ ] Compare Phase 2 temperature adjustments
- [ ] Create comparison table: baseline vs Q-table results

### 3. Final Artifacts
- [ ] Archive Q-table validation logs
- [ ] Update `baseline_results_summary.txt` with Q-table comparison
- [ ] Update `docs/deep_coupling_integration_notes.md` with final results
- [ ] Create merge request with summary

---

## 📋 Tuning Branch Integration Tasks

### For Tuning Branch Team

#### 1. Review & Understanding
- [ ] Pull `fluxnet-v2-feature` branch
- [ ] Read `docs/deep_coupling_integration_notes.md`
- [ ] Understand `[metaphysical_coupling]` config section
- [ ] Review modified files list (Section 10 in integration notes)
- [ ] Understand geometry reward shaping mechanism

#### 2. Configuration Integration
- [ ] Copy `configs/dsjc250_deep_coupling.toml` as template
- [ ] Merge `[metaphysical_coupling]` section into tuning configs
- [ ] Add `--fluxnet-qtable` flag to tuning runs
- [ ] Ensure `enable_telemetry = true` for stress tracking

#### 3. Baseline Tests (Without Coupling)
- [ ] Run tuning branch's best parameter set WITHOUT coupling
- [ ] Record chromatic number, runtime, convergence
- [ ] Establish baseline for comparison

#### 4. Coupling Tests (With Coupling)
- [ ] Run tuning branch's best parameters WITH coupling enabled
- [ ] Use same parameters, only toggle `[metaphysical_coupling] enabled = true`
- [ ] Record chromatic number, stress metrics, reward logs

#### 5. Q-Table Tests (Coupling + Retrained RL)
- [ ] Run tuning branch's best parameters WITH coupling + Q-table
- [ ] Use `--fluxnet-qtable artifacts/fluxnet/curriculum_bank_v3_geometry.bin`
- [ ] Record chromatic number, RL reward logs, stress trajectory

#### 6. Comparative Analysis
- [ ] Compare chromatic numbers: baseline → coupling → coupling+Q-table
- [ ] Measure stress reduction across configurations
- [ ] Analyze RL reward bonuses (if any)
- [ ] Determine if coupling provides measurable improvement

#### 7. Ablation Studies (Optional)
- [ ] Test with `enable_early_phase_seeding = false`
- [ ] Test with `enable_reward_shaping = false`
- [ ] Test with different `reward_shaping_scale` values
- [ ] Determine which coupling features provide most value

---

## 🔀 Merge Decision Criteria

### Merge to Main IF:
- ✅ Coupling improves chromatic number over baseline
- ✅ Stress telemetry shows measurable reduction
- ✅ No performance regressions (runtime acceptable)
- ✅ RL reward shaping demonstrates learning
- ✅ Integration with tuning params is successful

### Keep as Feature Flag IF:
- ⚠️ Marginal improvement (<2% chromatic reduction)
- ⚠️ High computational overhead (>50% runtime increase)
- ⚠️ Inconsistent results across different graphs
- ⚠️ Requires more tuning/investigation

### Defer Merge IF:
- ❌ No improvement or degradation
- ❌ Unstable results or crashes
- ❌ Conflicts with tuning branch changes
- ❌ Needs more development (telemetry export, etc.)

---

## 📁 Critical Files for Integration

### Configuration
```
configs/dsjc250_deep_coupling.toml
```

### Artifacts
```
artifacts/fluxnet/curriculum_bank_v3_geometry.bin  (Q-table - IN PROGRESS)
artifacts/fluxnet/baseline_results_summary.txt     (Baseline results)
artifacts/logs/baseline_gpu_16attempts_noqtable.log (Full baseline log)
```

### Documentation
```
docs/deep_coupling_integration_notes.md            (Integration guide)
artifacts/fluxnet/README.md                        (Training params)
INTEGRATION_CHECKLIST.md                           (This file)
DEEP_COUPLING_IMPLEMENTATION.md                    (Technical details)
```

### Code (Review These)
```
prism-fluxnet/src/core/state.rs                    (Geometry metrics)
prism-fluxnet/src/core/controller.rs               (Reward shaping)
prism-pipeline/src/orchestrator/mod.rs             (Geometry propagation)
prism-pipeline/src/config/mod.rs                   (Coupling config)
prism-phases/src/phase1_active_inference.rs        (Early seeding)
prism-phases/src/phase2_thermodynamic.rs           (Temp adjustment)
```

---

## 🐛 Known Issues & Workarounds

### 1. Telemetry JSONL Export Not Working
**Issue**: Config specifies `telemetry_path` but file not created

**Workaround**: Use `--enable-profiler --profiler-output <path>`

**Action**: May need to implement JSONL writer in orchestrator

### 2. Active Inference PTX Missing
**Issue**: Phase 1 GPU init fails, falls back to CPU

**Impact**: Minimal (Phase 1 is <1ms on CPU)

**Action**: Low priority

### 3. Geometry Reward Bonus Not Logged
**Issue**: No "FluxNet: Geometry reward bonus" messages

**Cause**: Stress delta is zero for this graph (stress constant at 0.533)

**Expected**: With Q-table, RL may learn policies that change stress

---

## 📞 Contact & Questions

For questions about deep coupling integration:
1. **Integration Guide**: `docs/deep_coupling_integration_notes.md`
2. **Technical Details**: `DEEP_COUPLING_IMPLEMENTATION.md`
3. **Code Comments**: See `prism-fluxnet/src/core/` for RL implementation

---

## ⏱️ Timeline Estimate

### Training Completion (Current)
- **Q-Table Training**: 2-4 hours remaining
- **Status**: Epoch 3/1000 (as of last check)

### Validation Runs (After Training)
- **16 attempts**: ~12 minutes (similar to baseline)
- **128 attempts**: ~90-100 minutes
- **Total validation**: ~2 hours

### Tuning Branch Integration (Estimated)
- **Review & setup**: 1-2 hours
- **Baseline tests**: 30-60 minutes
- **Coupling tests**: 30-60 minutes
- **Q-table tests**: 30-60 minutes
- **Analysis**: 1-2 hours
- **Total**: ~4-7 hours

---

## ✅ Final Deliverables Checklist

**Ready for Handoff**:
- [x] Integration documentation complete
- [x] Baseline results archived
- [x] Q-table training started
- [x] Artifacts directory structured
- [x] Configuration files ready
- [x] Integration checklist created

**Awaiting Training Completion**:
- [ ] Q-table binary file
- [ ] Training log final results
- [ ] Validation runs (16 & 128 attempts)
- [ ] Comparative analysis
- [ ] Final result summary

**Estimated Handoff**: ~4 hours from now (pending training completion)

---

**End of Integration Checklist**
