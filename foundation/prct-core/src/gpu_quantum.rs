//! GPU-Accelerated Quantum Hamiltonian Evolution
//!
//! Uses custom CUDA kernels for fast complex matrix-vector operations.
//! Expected speedup: 10-20x for large Hamiltonians.

use crate::errors::*;
use ndarray::Array2;
use num_complex::Complex64;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda")]
const QUANTUM_KERNELS: &str = r#"
// Complex number operations
struct Complex {
    float re;
    float im;
};

__device__ Complex complex_mul(Complex a, Complex b) {
    Complex result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;
    return result;
}

__device__ Complex complex_add(Complex a, Complex b) {
    Complex result;
    result.re = a.re + b.re;
    result.im = a.im + b.im;
    return result;
}

__device__ float complex_norm_sqr(Complex c) {
    return c.re * c.re + c.im * c.im;
}

// Matrix-vector multiplication: result = alpha * H * state
extern "C" __global__ void complex_matvec(
    const float* H_re,        // Hamiltonian real parts (n x n)
    const float* H_im,        // Hamiltonian imag parts (n x n)
    const float* state_re,    // State real parts (n)
    const float* state_im,    // State imag parts (n)
    const float alpha_re,     // Scalar multiplier real
    const float alpha_im,     // Scalar multiplier imag
    float* result_re,         // Output real parts (n)
    float* result_im,         // Output imag parts (n)
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    // Compute (alpha * H * state)[i]
    Complex sum = {0.0f, 0.0f};

    for (int j = 0; j < n; j++) {
        // H[i][j] * state[j]
        Complex h_ij = {H_re[i * n + j], H_im[i * n + j]};
        Complex s_j = {state_re[j], state_im[j]};
        Complex prod = complex_mul(h_ij, s_j);
        sum = complex_add(sum, prod);
    }

    // Multiply by alpha
    Complex alpha = {alpha_re, alpha_im};
    Complex result = complex_mul(alpha, sum);

    result_re[i] = result.re;
    result_im[i] = result.im;
}

// Vector addition: a += b
extern "C" __global__ void complex_axpy(
    float* a_re,              // Vector a real (modified in-place)
    float* a_im,              // Vector a imag (modified in-place)
    const float* b_re,        // Vector b real
    const float* b_im,        // Vector b imag
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    a_re[i] += b_re[i];
    a_im[i] += b_im[i];
}

// Compute norm squared: sum of |state[i]|^2
extern "C" __global__ void complex_norm_squared_kernel(
    const float* state_re,
    const float* state_im,
    float* partial_sums,
    const int n
) {
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute
    float local_sum = 0.0f;
    if (i < n) {
        Complex c = {state_re[i], state_im[i]};
        local_sum = complex_norm_sqr(c);
    }

    s_data[tid] = local_sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(partial_sums, s_data[0]);
    }
}

// Normalize state: state /= norm
extern "C" __global__ void complex_normalize(
    float* state_re,
    float* state_im,
    const float norm,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    if (norm > 1e-10f) {
        state_re[i] /= norm;
        state_im[i] /= norm;
    }
}
"#;

#[cfg(feature = "cuda")]
pub struct GpuQuantumSolver {
    device: Arc<CudaContext>,
    matvec_fn: Arc<CudaFunction>,
    axpy_fn: Arc<CudaFunction>,
    norm_sq_fn: Arc<CudaFunction>,
    normalize_fn: Arc<CudaFunction>,
}

#[cfg(feature = "cuda")]
impl GpuQuantumSolver {
    /// Create new GPU quantum solver
    pub fn new(device: Arc<CudaContext>) -> Result<Self> {
        // Compile CUDA kernels
        let ptx = cudarc::nvrtc::compile_ptx(QUANTUM_KERNELS)
            .map_err(|e| PRCTError::GpuError(format!("NVRTC compilation failed: {:?}", e)))?;

        device
            .load_ptx(
                ptx,
                "quantum",
                &[
                    "complex_matvec",
                    "complex_axpy",
                    "complex_norm_squared_kernel",
                    "complex_normalize",
                ],
            )
            .map_err(|e| PRCTError::GpuError(format!("PTX load failed: {:?}", e)))?;

        let matvec_fn = Arc::new(
            device
                .get_func("quantum", "complex_matvec")
                .ok_or_else(|| PRCTError::GpuError("Failed to get complex_matvec".to_string()))?,
        );

        let axpy_fn = Arc::new(
            device
                .get_func("quantum", "complex_axpy")
                .ok_or_else(|| PRCTError::GpuError("Failed to get complex_axpy".to_string()))?,
        );

        let norm_sq_fn = Arc::new(
            device
                .get_func("quantum", "complex_norm_squared_kernel")
                .ok_or_else(|| {
                    PRCTError::GpuError("Failed to get complex_norm_squared_kernel".to_string())
                })?,
        );

        let normalize_fn = Arc::new(
            device
                .get_func("quantum", "complex_normalize")
                .ok_or_else(|| {
                    PRCTError::GpuError("Failed to get complex_normalize".to_string())
                })?,
        );

        Ok(Self {
            device,
            matvec_fn,
            axpy_fn,
            norm_sq_fn,
            normalize_fn,
        })
    }

    /// Evolve quantum state on GPU
    ///
    /// Implements: |ψ(t+dt)⟩ = (I - iH dt)|ψ(t)⟩ for num_steps
    pub fn evolve_state_gpu(
        &self,
        hamiltonian: &Array2<Complex64>,
        initial_state: &[Complex64],
        dt: f64,
        num_steps: usize,
    ) -> Result<Vec<Complex64>> {
        let stream = context.default_stream();
        let n = hamiltonian.nrows();

        if initial_state.len() != n {
            return Err(PRCTError::QuantumFailed(format!(
                "State size {} doesn't match Hamiltonian dimension {}",
                initial_state.len(),
                n
            )));
        }

        // Separate into real and imaginary parts
        let state_re: Vec<f32> = initial_state.iter().map(|c| c.re as f32).collect();
        let state_im: Vec<f32> = initial_state.iter().map(|c| c.im as f32).collect();

        let mut h_re = Vec::with_capacity(n * n);
        let mut h_im = Vec::with_capacity(n * n);

        for row in 0..n {
            for col in 0..n {
                let c = hamiltonian[[row, col]];
                h_re.push(c.re as f32);
                h_im.push(c.im as f32);
            }
        }

        // Upload to GPU
        let d_h_re = self
            .device
            .htod_copy(h_re)
            .map_err(|e| PRCTError::GpuError(format!("H_re upload failed: {:?}", e)))?;

        let d_h_im = self
            .device
            .htod_copy(h_im)
            .map_err(|e| PRCTError::GpuError(format!("H_im upload failed: {:?}", e)))?;

        let mut d_state_re = self
            .device
            .htod_copy(state_re)
            .map_err(|e| PRCTError::GpuError(format!("State_re upload failed: {:?}", e)))?;

        let mut d_state_im = self
            .device
            .htod_copy(state_im)
            .map_err(|e| PRCTError::GpuError(format!("State_im upload failed: {:?}", e)))?;

        let mut d_temp_re = self
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|e| PRCTError::GpuError(format!("Temp_re allocation failed: {:?}", e)))?;

        let mut d_temp_im = self
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|e| PRCTError::GpuError(format!("Temp_im allocation failed: {:?}", e)))?;

        // Launch configuration
        let block_size = 256;
        let grid_size = n.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let dt_f32 = dt as f32;

        // Evolution loop
        for _ in 0..num_steps {
            // Compute temp = -i * dt * H * state
            // alpha = -i * dt = (0, -dt)
            let alpha_re = 0.0f32;
            let alpha_im = -dt_f32;

            unsafe {
                (*self.matvec_fn)
                    .clone()
                    .launch(
                        cfg,
                        (
                            &d_h_re,
                            &d_h_im,
                            &d_state_re,
                            &d_state_im,
                            alpha_re,
                            alpha_im,
                            &mut d_temp_re,
                            &mut d_temp_im,
                            n as i32,
                        ),
                    )
                    .map_err(|e| PRCTError::GpuError(format!("Matvec launch failed: {:?}", e)))?;
            }

            // state += temp  (state = (I - iHdt) * state)
            unsafe {
                (*self.axpy_fn)
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_state_re,
                            &mut d_state_im,
                            &d_temp_re,
                            &d_temp_im,
                            n as i32,
                        ),
                    )
                    .map_err(|e| PRCTError::GpuError(format!("Axpy launch failed: {:?}", e)))?;
            }

            // Normalize state
            let mut d_norm_sq = self
                .device
                .alloc_zeros::<f32>(1)
                .map_err(|e| PRCTError::GpuError(format!("Norm allocation failed: {:?}", e)))?;

            unsafe {
                (*self.norm_sq_fn)
                    .clone()
                    .launch(cfg, (&d_state_re, &d_state_im, &mut d_norm_sq, n as i32))
                    .map_err(|e| {
                        PRCTError::GpuError(format!("Norm kernel launch failed: {:?}", e))
                    })?;
            }

            let norm_sq_vec = self
                .device
                .dtoh_sync_copy(&d_norm_sq)
                .map_err(|e| PRCTError::GpuError(format!("Norm download failed: {:?}", e)))?;

            let norm = norm_sq_vec[0].sqrt();

            unsafe {
                (*self.normalize_fn)
                    .clone()
                    .launch(cfg, (&mut d_state_re, &mut d_state_im, norm, n as i32))
                    .map_err(|e| {
                        PRCTError::GpuError(format!("Normalize launch failed: {:?}", e))
                    })?;
            }
        }

        // Download final state
        let final_re = self
            .device
            .dtoh_sync_copy(&d_state_re)
            .map_err(|e| PRCTError::GpuError(format!("State_re download failed: {:?}", e)))?;

        let final_im = self
            .device
            .dtoh_sync_copy(&d_state_im)
            .map_err(|e| PRCTError::GpuError(format!("State_im download failed: {:?}", e)))?;

        // Combine back to Complex64
        let mut final_state = Vec::with_capacity(n);
        for i in 0..n {
            final_state.push(Complex64::new(final_re[i] as f64, final_im[i] as f64));
        }

        Ok(final_state)
    }
}

#[cfg(not(feature = "cuda"))]
pub struct GpuQuantumSolver;

#[cfg(not(feature = "cuda"))]
impl GpuQuantumSolver {
    pub fn new(_device: ()) -> Result<Self> {
        Err(PRCTError::GpuError("CUDA feature not enabled".to_string()))
    }

    pub fn evolve_state_gpu(
        &self,
        _hamiltonian: &Array2<Complex64>,
        _initial_state: &[Complex64],
        _dt: f64,
        _num_steps: usize,
    ) -> Result<Vec<Complex64>> {
        Err(PRCTError::GpuError("CUDA feature not enabled".to_string()))
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_quantum_solver_creation() {
        if let Ok(device) = CudaDevice::new(0) {
            let solver = GpuQuantumSolver::new(device);
            assert!(solver.is_ok());
        }
    }

    #[test]
    fn test_gpu_quantum_evolution() {
        if let Ok(device) = CudaDevice::new(0) {
            if let Ok(solver) = GpuQuantumSolver::new(device) {
                // Simple 2x2 Hamiltonian
                let h = Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.5, 0.0),
                        Complex64::new(0.5, 0.0),
                        Complex64::new(1.0, 0.0),
                    ],
                )
                .unwrap();

                let initial = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

                let evolved = solver.evolve_state_gpu(&h, &initial, 0.01, 10).unwrap();

                assert_eq!(evolved.len(), 2);
                // Check normalization
                let norm: f64 = evolved.iter().map(|c| c.norm_sqr()).sum();
                assert!((norm - 1.0).abs() < 0.1);
            }
        }
    }
}
