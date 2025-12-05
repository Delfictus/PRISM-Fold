# PRISM-LBS GPU-FUSED METRICS IMPLEMENTATION BLUEPRINT v2.0 FINAL
Target: RTX 3060 12GB | 500-Structure Batches | Ground Truth as First-Class GPU Tensor
Memory Layout: Global atom pool with pre-adjusted ca_indices + residue-offset indexing

## FILES TO MODIFY (3 files only)
1. `crates/prism-gpu/src/kernels/mega_fused_pocket_kernel.cu`
2. `crates/prism-gpu/src/mega_fused_batch.rs`
3. `crates/prism-lbs/src/lib.rs`

## MANDATORY FINAL FIXES (2025-12-05)

1. StructureOffset upload: Use `bytemuck::cast_slice` or direct typed upload
2. Metrics output buffer: `alloc_zeros::<BatchMetricsOutput>(n_structures)` NOT `u8`
3. GT mask validation: `assert_eq!(gt.len(), n_residues)`
4. Binary search: `prefix[mid] <= tile_id` → `return low - 1`
5. Return real BatchStructureOutput (not empty vec)

---

## EDIT 1: mega_fused_pocket_kernel.cu

### 1A. Structures (INSERT AFTER line 173, after d_default_params)

```cuda
//=============================================================================
// BATCH METRICS STRUCTURES
//=============================================================================

#define N_BINS 100

struct __align__(8) StructureOffset {
    int structure_id;
    int residue_start;
    int residue_count;
    int padding;
};

struct __align__(16) BatchMetricsOutput {
    int structure_id;
    int n_residues;
    int true_positives;
    int false_positives;
    int true_negatives;
    int false_negatives;
    float precision;
    float recall;
    float f1_score;
    float mcc;
    float auc_roc;
    float auprc;
    float avg_druggability;
    int n_pockets_detected;
};

__device__ __forceinline__ int find_structure_id(
    const int* __restrict__ prefix,
    int n_structures,
    int tile_id
) {
    int low = 0;
    int high = n_structures;
    while (low < high) {
        int mid = (low + high) >> 1;
        if (prefix[mid] <= tile_id) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low - 1;
}

__device__ __forceinline__ int get_bin(float score) {
    return min(N_BINS - 1, max(0, (int)(score * N_BINS)));
}
```

### 1B. Stage 7 (INSERT AFTER stage6)

```cuda
//=============================================================================
// STAGE 7: GPU-FUSED METRICS + HISTOGRAM COLLECTION
//=============================================================================

__device__ void stage7_compute_metrics(
    int n_residues,
    int tile_idx,
    const unsigned char* __restrict__ gt_pocket_mask,
    MegaFusedSharedMemory* smem,
    const MegaFusedParams* params,
    int* tp_out, int* fp_out, int* tn_out, int* fn_out,
    float* score_sum_out, int* pocket_count_out,
    unsigned long long* hist_pos,
    unsigned long long* hist_neg
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);
    if (active) {
        int predicted = smem->pocket_assignment[local_idx];
        int actual = (int)gt_pocket_mask[global_idx];
        float score = smem->consensus_score[local_idx];

        if (predicted == 1 && actual == 1) atomicAdd(tp_out, 1);
        else if (predicted == 1 && actual == 0) atomicAdd(fp_out, 1);
        else if (predicted == 0 && actual == 0) atomicAdd(tn_out, 1);
        else if (predicted == 0 && actual == 1) atomicAdd(fn_out, 1);

        if (predicted == 1) {
            atomicAdd(score_sum_out, score);
            atomicAdd(pocket_count_out, 1);
        }

        int bin = get_bin(score);
        if (actual == 1) atomicAdd(&hist_pos[bin], 1ULL);
        else atomicAdd(&hist_neg[bin], 1ULL);
    }
    __syncthreads();
}
```

### 1C. Batch Kernel (INSERT AFTER host launchers)

```cuda
//=============================================================================
// BATCH KERNEL WITH GROUND TRUTH + METRICS
//=============================================================================

extern "C" __global__ void __launch_bounds__(256, 4)
mega_fused_pocket_detection_batch_with_metrics(
    const float* __restrict__ atoms_flat,
    const int* __restrict__ ca_indices_flat,
    const float* __restrict__ conservation_flat,
    const float* __restrict__ bfactor_flat,
    const float* __restrict__ burial_flat,
    const unsigned char* __restrict__ gt_pocket_mask_flat,
    const StructureOffset* __restrict__ offsets,
    const int* __restrict__ tile_prefix_sum,
    int n_structures,
    int total_tiles,
    float* __restrict__ consensus_scores_flat,
    int* __restrict__ confidence_flat,
    int* __restrict__ signal_mask_flat,
    int* __restrict__ pocket_assignment_flat,
    float* __restrict__ centrality_flat,
    int* __restrict__ tp_counts,
    int* __restrict__ fp_counts,
    int* __restrict__ tn_counts,
    int* __restrict__ fn_counts,
    float* __restrict__ score_sums,
    int* __restrict__ pocket_counts,
    unsigned long long* __restrict__ hist_pos_flat,
    unsigned long long* __restrict__ hist_neg_flat,
    const MegaFusedParams* __restrict__ params
) {
    if (blockIdx.x >= total_tiles) return;
    int sid = find_structure_id(tile_prefix_sum, n_structures, blockIdx.x);

    __shared__ MegaFusedSharedMemory smem;
    __shared__ int s_residue_offset, s_n_residues, s_tile_idx;
    if (threadIdx.x == 0) {
        s_residue_offset = offsets[sid].residue_start;
        s_n_residues = offsets[sid].residue_count;
        s_tile_idx = blockIdx.x - tile_prefix_sum[sid];
    }
    __syncthreads();

    int residue_offset = s_residue_offset;
    int n_residues = s_n_residues;
    int tile_idx = s_tile_idx;
    int local_idx = threadIdx.x;

    if (local_idx < TILE_SIZE) {
        smem.reservoir_state[local_idx][0] = 0.0f;
        smem.pocket_assignment[local_idx] = 0;
    }
    __syncthreads();

    const float* atoms = atoms_flat;
    const int* ca_indices = ca_indices_flat + residue_offset;
    const float* conservation = conservation_flat + residue_offset;
    const float* bfactor = bfactor_flat + residue_offset;
    const float* burial = burial_flat + residue_offset;
    const unsigned char* gt_mask = gt_pocket_mask_flat + residue_offset;
    float* consensus_out = consensus_scores_flat + residue_offset;
    int* confidence_out = confidence_flat + residue_offset;
    int* signal_out = signal_mask_flat + residue_offset;
    int* pocket_out = pocket_assignment_flat + residue_offset;
    float* centrality_out_ptr = centrality_flat + residue_offset;

    stage1_distance_contact(atoms, ca_indices, n_residues, tile_idx, tile_idx, &smem, params);
    stage2_local_features(conservation, bfactor, burial, n_residues, tile_idx, &smem);
    stage3_network_centrality(n_residues, tile_idx, &smem, params);
    stage4_dendritic_reservoir(n_residues, tile_idx, &smem, params);
    stage5_consensus(n_residues, tile_idx, &smem, params);
    stage6_kempe_refinement(n_residues, tile_idx, &smem, params);

    stage7_compute_metrics(
        n_residues, tile_idx, gt_mask, &smem, params,
        &tp_counts[sid], &fp_counts[sid], &tn_counts[sid], &fn_counts[sid],
        &score_sums[sid], &pocket_counts[sid],
        hist_pos_flat + sid * N_BINS,
        hist_neg_flat + sid * N_BINS
    );

    int out_idx = tile_idx * TILE_SIZE + local_idx;
    if (local_idx < TILE_SIZE && out_idx < n_residues) {
        consensus_out[out_idx] = smem.consensus_score[local_idx];
        confidence_out[out_idx] = smem.confidence[local_idx];
        signal_out[out_idx] = smem.signal_mask[local_idx];
        pocket_out[out_idx] = smem.pocket_assignment[local_idx];
        centrality_out_ptr[out_idx] = smem.centrality[local_idx];
    }
}
```

### 1D. Finalize Kernel

```cuda
//=============================================================================
// FINALIZE METRICS KERNEL - REAL AUC-ROC & AUPRC
//=============================================================================

extern "C" __global__ void finalize_batch_metrics(
    const int* __restrict__ tp_counts,
    const int* __restrict__ fp_counts,
    const int* __restrict__ tn_counts,
    const int* __restrict__ fn_counts,
    const float* __restrict__ score_sums,
    const int* __restrict__ pocket_counts,
    const unsigned long long* __restrict__ hist_pos,
    const unsigned long long* __restrict__ hist_neg,
    BatchMetricsOutput* __restrict__ metrics_out,
    const StructureOffset* __restrict__ offsets,
    int n_structures
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_structures) return;

    const unsigned long long* pos = hist_pos + sid * N_BINS;
    const unsigned long long* neg = hist_neg + sid * N_BINS;

    int tp = tp_counts[sid];
    int fp = fp_counts[sid];
    int tn = tn_counts[sid];
    int fn = fn_counts[sid];
    float score_sum = score_sums[sid];
    int n_pockets = pocket_counts[sid];

    float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
    float recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
    float f1 = (precision + recall > 0.0f) ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    float mcc_num = (float)(tp * tn - fp * fn);
    float denom = sqrtf(fmaxf(1.0f, (float)(tp + fp)) *
                        fmaxf(1.0f, (float)(tp + fn)) *
                        fmaxf(1.0f, (float)(tn + fp)) *
                        fmaxf(1.0f, (float)(tn + fn)));
    float mcc = mcc_num / denom;

    float avg_drug = (n_pockets > 0) ? score_sum / n_pockets : 0.0f;

    float auc_roc = 0.0f;
    float prev_tpr = 0.0f, prev_fpr = 0.0f;
    unsigned long long total_pos = 0, total_neg = 0;
    for (int i = 0; i < N_BINS; ++i) {
        total_pos += pos[i];
        total_neg += neg[i];
    }
    float inv_pos = (total_pos > 0) ? 1.0f / total_pos : 0.0f;
    float inv_neg = (total_neg > 0) ? 1.0f / total_neg : 0.0f;

    for (int i = N_BINS - 1; i >= 0; --i) {
        float tpr = prev_tpr + (float)pos[i] * inv_pos;
        float fpr = prev_fpr + (float)neg[i] * inv_neg;
        auc_roc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5f;
        prev_tpr = tpr; prev_fpr = fpr;
    }

    float auprc = 0.0f;
    float prev_prec = 1.0f, prev_rec = 0.0f;
    unsigned long long cum_tp = 0, cum_fp = 0;
    for (int i = N_BINS - 1; i >= 0; --i) {
        cum_tp += pos[i];
        cum_fp += neg[i];
        float cur_prec = (cum_tp + cum_fp > 0) ? (float)cum_tp / (cum_tp + cum_fp) : 1.0f;
        float cur_rec = (float)cum_tp * inv_pos;
        auprc += (cur_rec - prev_rec) * (cur_prec + prev_prec) * 0.5f;
        prev_prec = cur_prec; prev_rec = cur_rec;
    }

    metrics_out[sid].structure_id = offsets[sid].structure_id;
    metrics_out[sid].n_residues = offsets[sid].residue_count;
    metrics_out[sid].true_positives = tp;
    metrics_out[sid].false_positives = fp;
    metrics_out[sid].true_negatives = tn;
    metrics_out[sid].false_negatives = fn;
    metrics_out[sid].precision = precision;
    metrics_out[sid].recall = recall;
    metrics_out[sid].f1_score = f1;
    metrics_out[sid].mcc = mcc;
    metrics_out[sid].auc_roc = auc_roc;
    metrics_out[sid].auprc = auprc;
    metrics_out[sid].avg_druggability = avg_drug;
    metrics_out[sid].n_pockets_detected = n_pockets;
}
```

---

## EDIT 2: mega_fused_batch.rs

### 2A. Structures (INSERT after StructureInput)

```rust
/// Structure input WITH ground truth for validation batches
#[derive(Debug, Clone)]
pub struct StructureInputWithGT {
    pub base: StructureInput,
    pub gt_pocket_mask: Vec<u8>,
}

/// Structure offset for batch mapping - MUST match CUDA StructureOffset
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StructureOffset {
    pub structure_id: i32,
    pub residue_start: i32,
    pub residue_count: i32,
    pub padding: i32,
}

/// Per-structure metrics from GPU - MUST match CUDA BatchMetricsOutput
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchMetricsOutput {
    pub structure_id: i32,
    pub n_residues: i32,
    pub true_positives: i32,
    pub false_positives: i32,
    pub true_negatives: i32,
    pub false_negatives: i32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub mcc: f32,
    pub auc_roc: f32,
    pub auprc: f32,
    pub avg_druggability: f32,
    pub n_pockets_detected: i32,
}

const N_BINS: usize = 100;
```

### 2B. PackedBatchWithGT

```rust
#[derive(Debug)]
pub struct PackedBatchWithGT {
    pub base: PackedBatch,
    pub gt_pocket_mask_packed: Vec<u8>,
    pub offsets: Vec<StructureOffset>,
    pub tile_prefix_sum: Vec<i32>,
    pub total_tiles: i32,
}

impl PackedBatchWithGT {
    pub fn from_structures_with_gt(structures: &[StructureInputWithGT]) -> Result<Self, PrismError> {
        // FIX 3: GT mask validation
        for s in structures {
            assert_eq!(
                s.gt_pocket_mask.len(),
                s.base.n_residues(),
                "GT mask length mismatch for structure {}",
                s.base.id
            );
        }

        let base_inputs: Vec<StructureInput> = structures.iter().map(|s| s.base.clone()).collect();
        let base = PackedBatch::from_structures(&base_inputs)?;

        let mut gt_packed: Vec<u8> = Vec::with_capacity(base.total_residues);
        let mut offsets: Vec<StructureOffset> = Vec::with_capacity(structures.len());
        let mut tile_prefix_sum: Vec<i32> = vec![0; structures.len() + 1];

        let mut residue_offset = 0i32;
        let tile_size = 32i32;

        for (idx, s) in structures.iter().enumerate() {
            let n_res = s.base.n_residues() as i32;
            let n_tiles = (n_res + tile_size - 1) / tile_size;

            offsets.push(StructureOffset {
                structure_id: idx as i32,
                residue_start: residue_offset,
                residue_count: n_res,
                padding: 0,
            });

            tile_prefix_sum[idx + 1] = tile_prefix_sum[idx] + n_tiles;
            gt_packed.extend_from_slice(&s.gt_pocket_mask);
            residue_offset += n_res;
        }

        let total_tiles = tile_prefix_sum[structures.len()];

        Ok(Self {
            base,
            gt_pocket_mask_packed: gt_packed,
            offsets,
            tile_prefix_sum,
            total_tiles,
        })
    }
}
```

### 2C. Output Structures

```rust
#[derive(Debug)]
pub struct BatchOutputWithMetrics {
    pub structures: Vec<BatchStructureOutput>,
    pub metrics: Vec<BatchMetricsOutput>,
    pub kernel_time_us: u64,
    pub aggregate: AggregateMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct AggregateMetrics {
    pub mean_f1: f32,
    pub mean_mcc: f32,
    pub mean_auc_roc: f32,
    pub mean_auprc: f32,
    pub mean_precision: f32,
    pub mean_recall: f32,
}
```

### 2D. detect_pockets_batch_with_metrics (with ALL FIXES)

```rust
pub fn detect_pockets_batch_with_metrics(
    &mut self,
    batch: &PackedBatchWithGT,
    config: &MegaFusedConfig,
) -> Result<BatchOutputWithMetrics, PrismError> {
    let n_structures = batch.offsets.len();
    let total_residues = batch.base.total_residues;
    let total_tiles = batch.total_tiles as u32;

    let start = Instant::now();

    // Upload inputs
    let d_atoms = self.stream.memcpy_stod(&batch.base.atoms_packed)?;
    let d_ca_indices = self.stream.memcpy_stod(&batch.base.ca_indices_packed)?;
    let d_conservation = self.stream.memcpy_stod(&batch.base.conservation_packed)?;
    let d_bfactor = self.stream.memcpy_stod(&batch.base.bfactor_packed)?;
    let d_burial = self.stream.memcpy_stod(&batch.base.burial_packed)?;
    let d_gt_mask = self.stream.memcpy_stod(&batch.gt_pocket_mask_packed)?;

    // FIX 1: Direct typed upload using bytemuck
    let d_offsets = self.stream.memcpy_stod(bytemuck::cast_slice(&batch.offsets))?;
    let d_tile_prefix = self.stream.memcpy_stod(&batch.tile_prefix_sum)?;

    // Output buffers
    let d_consensus = self.stream.alloc_zeros::<f32>(total_residues)?;
    let d_confidence = self.stream.alloc_zeros::<i32>(total_residues)?;
    let d_signal_mask = self.stream.alloc_zeros::<i32>(total_residues)?;
    let d_pocket_assignment = self.stream.alloc_zeros::<i32>(total_residues)?;
    let d_centrality = self.stream.alloc_zeros::<f32>(total_residues)?;

    // Metric accumulators
    let d_tp = self.stream.alloc_zeros::<i32>(n_structures)?;
    let d_fp = self.stream.alloc_zeros::<i32>(n_structures)?;
    let d_tn = self.stream.alloc_zeros::<i32>(n_structures)?;
    let d_fn = self.stream.alloc_zeros::<i32>(n_structures)?;
    let d_score_sums = self.stream.alloc_zeros::<f32>(n_structures)?;
    let d_pocket_counts = self.stream.alloc_zeros::<i32>(n_structures)?;

    // Histograms
    let d_hist_pos = self.stream.alloc_zeros::<u64>(n_structures * N_BINS)?;
    let d_hist_neg = self.stream.alloc_zeros::<u64>(n_structures * N_BINS)?;

    // FIX 2: Typed metrics output buffer
    let d_metrics_out = self.stream.alloc_zeros::<BatchMetricsOutput>(n_structures)?;

    // Params
    let params = MegaFusedParams::from_config(config);
    let params_bytes: &[u8] = bytemuck::bytes_of(&params);
    let d_params = self.stream.memcpy_stod(params_bytes)?;

    // Launch main kernel
    let launch_config = LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_structures_i32 = n_structures as i32;
    let total_tiles_i32 = total_tiles as i32;

    unsafe {
        let mut builder = self.stream.launch_builder(&self.batch_metrics_func);
        builder.arg(&d_atoms);
        builder.arg(&d_ca_indices);
        builder.arg(&d_conservation);
        builder.arg(&d_bfactor);
        builder.arg(&d_burial);
        builder.arg(&d_gt_mask);
        builder.arg(&d_offsets);
        builder.arg(&d_tile_prefix);
        builder.arg(&n_structures_i32);
        builder.arg(&total_tiles_i32);
        builder.arg(&d_consensus);
        builder.arg(&d_confidence);
        builder.arg(&d_signal_mask);
        builder.arg(&d_pocket_assignment);
        builder.arg(&d_centrality);
        builder.arg(&d_tp);
        builder.arg(&d_fp);
        builder.arg(&d_tn);
        builder.arg(&d_fn);
        builder.arg(&d_score_sums);
        builder.arg(&d_pocket_counts);
        builder.arg(&d_hist_pos);
        builder.arg(&d_hist_neg);
        builder.arg(&d_params);
        builder.launch(launch_config)?;
    }

    // Launch finalize kernel
    let finalize_blocks = (n_structures + 127) / 128;
    let finalize_config = LaunchConfig {
        grid_dim: (finalize_blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = self.stream.launch_builder(&self.finalize_func);
        builder.arg(&d_tp);
        builder.arg(&d_fp);
        builder.arg(&d_tn);
        builder.arg(&d_fn);
        builder.arg(&d_score_sums);
        builder.arg(&d_pocket_counts);
        builder.arg(&d_hist_pos);
        builder.arg(&d_hist_neg);
        builder.arg(&d_metrics_out);
        builder.arg(&d_offsets);
        builder.arg(&n_structures_i32);
        builder.launch(finalize_config)?;
    }

    self.stream.synchronize()?;
    let kernel_time = start.elapsed();

    // Download metrics
    let metrics: Vec<BatchMetricsOutput> = self.stream.clone_dtoh(&d_metrics_out)?;

    // FIX 5: Download and build real BatchStructureOutput
    let consensus_all: Vec<f32> = self.stream.clone_dtoh(&d_consensus)?;
    let confidence_all: Vec<i32> = self.stream.clone_dtoh(&d_confidence)?;
    let signal_mask_all: Vec<i32> = self.stream.clone_dtoh(&d_signal_mask)?;
    let pocket_assignment_all: Vec<i32> = self.stream.clone_dtoh(&d_pocket_assignment)?;
    let centrality_all: Vec<f32> = self.stream.clone_dtoh(&d_centrality)?;

    let mut structures: Vec<BatchStructureOutput> = Vec::with_capacity(n_structures);
    for (idx, offset) in batch.offsets.iter().enumerate() {
        let start = offset.residue_start as usize;
        let end = start + offset.residue_count as usize;
        structures.push(BatchStructureOutput {
            id: batch.base.ids[idx].clone(),
            consensus_scores: consensus_all[start..end].to_vec(),
            confidence: confidence_all[start..end].to_vec(),
            signal_mask: signal_mask_all[start..end].to_vec(),
            pocket_assignment: pocket_assignment_all[start..end].to_vec(),
            centrality: centrality_all[start..end].to_vec(),
        });
    }

    // Compute aggregates
    let n = metrics.len() as f32;
    let aggregate = AggregateMetrics {
        mean_f1: metrics.iter().map(|m| m.f1_score).sum::<f32>() / n,
        mean_mcc: metrics.iter().map(|m| m.mcc).sum::<f32>() / n,
        mean_auc_roc: metrics.iter().map(|m| m.auc_roc).sum::<f32>() / n,
        mean_auprc: metrics.iter().map(|m| m.auprc).sum::<f32>() / n,
        mean_precision: metrics.iter().map(|m| m.precision).sum::<f32>() / n,
        mean_recall: metrics.iter().map(|m| m.recall).sum::<f32>() / n,
    };

    Ok(BatchOutputWithMetrics {
        structures,
        metrics,
        kernel_time_us: kernel_time.as_micros() as u64,
        aggregate,
    })
}
```

---

## EDIT 3: lib.rs

### Ground Truth Loader

```rust
pub fn load_cryptobench_ground_truth(
    dataset_path: &Path,
) -> Result<HashMap<String, Vec<usize>>> {
    let data: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(dataset_path)?
    )?;

    let mut gt_map: HashMap<String, Vec<usize>> = HashMap::new();

    if let Some(obj) = data.as_object() {
        for (pdb_id, entries) in obj {
            let mut residues: Vec<usize> = Vec::new();
            if let Some(arr) = entries.as_array() {
                for entry in arr {
                    if let Some(pocket_sel) = entry.get("apo_pocket_selection").and_then(|v| v.as_array()) {
                        for item in pocket_sel {
                            if let Some(s) = item.as_str() {
                                if let Some(res_num) = s.split('_').nth(1).and_then(|n| n.parse().ok()) {
                                    residues.push(res_num);
                                }
                            }
                        }
                    }
                }
            }
            gt_map.insert(pdb_id.to_lowercase(), residues);
        }
    }

    Ok(gt_map)
}
```

---

## PERFORMANCE EXPECTATIONS (RTX 3060 12GB)

| Metric | Target |
|--------|--------|
| 500 structures/batch | <120s total |
| Memory per structure | ~2.5MB |
| Stage 7 overhead | +6-8% |
| Finalize kernel | <50us |
| Metrics download | ~28KB |

## EXECUTION ORDER

1. Edit `mega_fused_pocket_kernel.cu`
2. Compile PTX: `nvcc --ptx -arch=sm_86 ...`
3. Edit `mega_fused_batch.rs`
4. Edit `lib.rs`
5. `cargo build --release`
6. Run 500-structure validation batch
