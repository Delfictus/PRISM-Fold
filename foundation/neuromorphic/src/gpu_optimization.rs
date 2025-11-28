//! GPU Performance Optimization and Analysis
//!
//! Advanced performance optimization utilities for RTX 5070 CUDA acceleration
//! Provides detailed performance hotspot analysis and optimization recommendations

use anyhow::Result;
use cudarc::driver::*;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;

/// Performance analysis metrics for CUDA operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operation_name: String,
    pub total_time_us: f64,
    pub gpu_compute_time_us: f64,
    pub memory_transfer_time_us: f64,
    pub memory_bandwidth_gb_s: f64,
    pub cuda_core_utilization: f64,
    pub occupancy_percentage: f64,
    pub call_count: u64,
}

/// Hot path identification and optimization recommendations
#[derive(Debug, Clone)]
pub struct HotspotAnalysis {
    pub hotspots: Vec<PerformanceHotspot>,
    pub total_analysis_time_ms: f64,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
}

#[derive(Debug, Clone)]
pub struct PerformanceHotspot {
    pub function_name: String,
    pub time_percentage: f64,
    pub absolute_time_us: f64,
    pub bottleneck_type: BottleneckType,
    pub optimization_priority: Priority,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    ComputeBound,    // Limited by GPU compute
    MemoryBound,     // Limited by memory bandwidth
    LaunchOverhead,  // CUDA kernel launch overhead
    CpuGpuTransfer,  // Host-device transfer bottleneck
    Synchronization, // Device synchronization overhead
}

#[derive(Debug, Clone)]
pub enum Priority {
    Critical, // >10% of total time
    High,     // 5-10% of total time
    Medium,   // 1-5% of total time
    Low,      // <1% of total time
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub description: String,
    pub expected_improvement: f64, // Percentage improvement
    pub implementation_difficulty: Difficulty,
    pub rtx_5070_specific: bool,
}

#[derive(Debug, Clone)]
pub enum Difficulty {
    Easy,   // Simple configuration change
    Medium, // Code refactoring required
    Hard,   // Major architectural changes
}

/// GPU performance profiler optimized for RTX 5070
pub struct GpuProfiler {
    device: Arc<CudaDevice>,
    metrics: DashMap<String, PerformanceMetrics>,
    profiling_start: Option<Instant>,
    enable_detailed_profiling: bool,
}

impl GpuProfiler {
    /// Create new GPU profiler
    pub fn new(device: Arc<CudaDevice>, enable_detailed: bool) -> Self {
        Self {
            device,
            metrics: DashMap::new(),
            profiling_start: None,
            enable_detailed_profiling: enable_detailed,
        }
    }

    /// Start profiling session
    pub fn start_session(&mut self) {
        self.profiling_start = Some(Instant::now());
        self.metrics.clear();
    }

    /// Profile a GPU operation
    pub fn profile_operation<F, R>(&self, operation_name: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        if !self.enable_detailed_profiling {
            return operation();
        }

        let start = Instant::now();

        // Execute operation and measure time (cudarc 0.9: use CPU timing)
        let result = operation()?;

        // Synchronize device to get accurate timing
        self.device.synchronize()?;

        let total_time = start.elapsed().as_micros() as f64;

        // Note: Without CUDA events, we use total time as GPU time estimate
        // This includes both compute and memory transfer
        let gpu_time = total_time;
        let memory_transfer_time: f64 = 0.0; // Cannot separate without events

        // Update metrics
        let mut metric = self
            .metrics
            .entry(operation_name.to_string())
            .or_insert_with(|| PerformanceMetrics {
                operation_name: operation_name.to_string(),
                total_time_us: 0.0,
                gpu_compute_time_us: 0.0,
                memory_transfer_time_us: 0.0,
                memory_bandwidth_gb_s: 0.0,
                cuda_core_utilization: 0.0,
                occupancy_percentage: 0.0,
                call_count: 0,
            });

        metric.total_time_us += total_time;
        metric.gpu_compute_time_us += gpu_time;
        metric.memory_transfer_time_us += memory_transfer_time.max(0.0_f64);
        metric.call_count += 1;

        Ok(result)
    }

    /// Analyze performance hotspots and generate optimization recommendations
    pub fn analyze_hotspots(&self) -> HotspotAnalysis {
        let analysis_start = Instant::now();

        // Calculate total time across all operations
        let total_time: f64 = self
            .metrics
            .iter()
            .map(|entry| entry.value().total_time_us)
            .sum();

        if total_time == 0.0 {
            return HotspotAnalysis {
                hotspots: vec![],
                total_analysis_time_ms: 0.0,
                optimization_recommendations: vec![],
            };
        }

        // Identify hotspots
        let mut hotspots: Vec<PerformanceHotspot> = self
            .metrics
            .iter()
            .map(|entry| {
                let metric = entry.value();
                let time_percentage = (metric.total_time_us / total_time) * 100.0;

                let bottleneck_type =
                    if metric.memory_transfer_time_us > metric.gpu_compute_time_us * 2.0 {
                        BottleneckType::CpuGpuTransfer
                    } else if metric.memory_transfer_time_us > metric.gpu_compute_time_us {
                        BottleneckType::MemoryBound
                    } else {
                        BottleneckType::ComputeBound
                    };

                let priority = if time_percentage > 10.0 {
                    Priority::Critical
                } else if time_percentage > 5.0 {
                    Priority::High
                } else if time_percentage > 1.0 {
                    Priority::Medium
                } else {
                    Priority::Low
                };

                PerformanceHotspot {
                    function_name: metric.operation_name.clone(),
                    time_percentage,
                    absolute_time_us: metric.total_time_us,
                    bottleneck_type,
                    optimization_priority: priority,
                }
            })
            .collect();

        // Sort by time percentage (highest first)
        hotspots.sort_by(|a, b| {
            b.time_percentage
                .partial_cmp(&a.time_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Generate optimization recommendations
        let recommendations = self.generate_optimization_recommendations(&hotspots);

        HotspotAnalysis {
            hotspots,
            total_analysis_time_ms: analysis_start.elapsed().as_millis() as f64,
            optimization_recommendations: recommendations,
        }
    }

    /// Generate RTX 5070-specific optimization recommendations
    fn generate_optimization_recommendations(
        &self,
        hotspots: &[PerformanceHotspot],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for hotspot in hotspots {
            match hotspot.bottleneck_type {
                BottleneckType::CpuGpuTransfer => {
                    recommendations.push(OptimizationRecommendation {
                        description: format!(
                            "Eliminate CPU-GPU transfers in '{}' by using GPU-only operations. \
                            Current overhead: {:.1}% of total time",
                            hotspot.function_name, hotspot.time_percentage
                        ),
                        expected_improvement: hotspot.time_percentage * 0.8, // 80% improvement potential
                        implementation_difficulty: Difficulty::Medium,
                        rtx_5070_specific: true,
                    });
                }
                BottleneckType::ComputeBound => {
                    recommendations.push(OptimizationRecommendation {
                        description: format!(
                            "Optimize CUDA kernel in '{}' for RTX 5070's 6,144 cores. \
                            Use larger block sizes (512 threads) and maximize occupancy",
                            hotspot.function_name
                        ),
                        expected_improvement: hotspot.time_percentage * 0.3, // 30% improvement potential
                        implementation_difficulty: Difficulty::Hard,
                        rtx_5070_specific: true,
                    });
                }
                BottleneckType::MemoryBound => {
                    recommendations.push(OptimizationRecommendation {
                        description: format!(
                            "Improve memory access patterns in '{}'. Use vectorized loads (float4) \
                            and leverage RTX 5070's high memory bandwidth",
                            hotspot.function_name
                        ),
                        expected_improvement: hotspot.time_percentage * 0.4, // 40% improvement potential
                        implementation_difficulty: Difficulty::Medium,
                        rtx_5070_specific: true,
                    });
                }
                BottleneckType::Synchronization => {
                    recommendations.push(OptimizationRecommendation {
                        description: format!(
                            "Reduce synchronization overhead in '{}' by using asynchronous operations \
                            and CUDA streams",
                            hotspot.function_name
                        ),
                        expected_improvement: hotspot.time_percentage * 0.6, // 60% improvement potential
                        implementation_difficulty: Difficulty::Easy,
                        rtx_5070_specific: false,
                    });
                }
                BottleneckType::LaunchOverhead => {
                    recommendations.push(OptimizationRecommendation {
                        description: format!(
                            "Reduce kernel launch overhead in '{}' by batching operations \
                            or using persistent kernels",
                            hotspot.function_name
                        ),
                        expected_improvement: hotspot.time_percentage * 0.5, // 50% improvement potential
                        implementation_difficulty: Difficulty::Medium,
                        rtx_5070_specific: false,
                    });
                }
            }
        }

        // Add general RTX 5070 optimization recommendations
        recommendations.push(OptimizationRecommendation {
            description: "Enable mixed precision (FP16) operations for 2x throughput improvement on RTX 5070's tensor cores".to_string(),
            expected_improvement: 25.0, // 25% overall improvement
            implementation_difficulty: Difficulty::Medium,
            rtx_5070_specific: true,
        });

        recommendations.push(OptimizationRecommendation {
            description: "Implement cooperative groups for better warp utilization on Ada Lovelace architecture".to_string(),
            expected_improvement: 15.0, // 15% improvement
            implementation_difficulty: Difficulty::Hard,
            rtx_5070_specific: true,
        });

        recommendations.push(OptimizationRecommendation {
            description: "Use cuBLAS tensor operations for matrix computations to leverage RTX 5070's hardware acceleration".to_string(),
            expected_improvement: 20.0, // 20% improvement
            implementation_difficulty: Difficulty::Easy,
            rtx_5070_specific: true,
        });

        recommendations
    }

    /// Generate detailed performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== RTX 5070 GPU Performance Analysis Report ===\n\n");

        let analysis = self.analyze_hotspots();

        report.push_str("🔥 PERFORMANCE HOTSPOTS:\n");
        for (i, hotspot) in analysis.hotspots.iter().enumerate() {
            report.push_str(&format!(
                "{}. {} ({:.1}% of total time, {:.2}ms)\n",
                i + 1,
                hotspot.function_name,
                hotspot.time_percentage,
                hotspot.absolute_time_us / 1000.0
            ));
            report.push_str(&format!(
                "   Type: {:?}, Priority: {:?}\n",
                hotspot.bottleneck_type, hotspot.optimization_priority
            ));
        }

        report.push_str("\n⚡ OPTIMIZATION RECOMMENDATIONS:\n");
        for (i, rec) in analysis.optimization_recommendations.iter().enumerate() {
            let rtx_icon = if rec.rtx_5070_specific {
                "🎯"
            } else {
                "💡"
            };
            report.push_str(&format!(
                "{}. {} {} (Expected improvement: {:.1}%, Difficulty: {:?})\n",
                i + 1,
                rtx_icon,
                rec.description,
                rec.expected_improvement,
                rec.implementation_difficulty
            ));
        }

        report.push_str(&format!(
            "\n📊 Analysis completed in {:.2}ms\n",
            analysis.total_analysis_time_ms
        ));

        report
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> Vec<PerformanceMetrics> {
        self.metrics
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }
}

/// RTX 5070-specific optimization utilities
pub struct Rtx5070Optimizer;

impl Rtx5070Optimizer {
    /// Get optimal CUDA kernel configuration for RTX 5070
    pub fn get_optimal_kernel_config(problem_size: usize) -> (u32, u32) {
        // RTX 5070 has 24 SMs, 6144 CUDA cores
        let block_size = if problem_size < 1000 {
            256 // Smaller problems benefit from smaller blocks
        } else {
            512 // Large problems can utilize full blocks
        };

        let grid_size = (problem_size as u32).div_ceil(block_size).min(4096);

        (block_size, grid_size)
    }

    /// Calculate memory bandwidth utilization
    pub fn calculate_memory_bandwidth_utilization(bytes_transferred: usize, time_us: f64) -> f64 {
        let bandwidth_gb_s =
            (bytes_transferred as f64) / (time_us * 1e-6) / (1024.0 * 1024.0 * 1024.0);
        let rtx_5070_peak_bandwidth = 504.0; // GB/s (theoretical peak)

        (bandwidth_gb_s / rtx_5070_peak_bandwidth) * 100.0
    }

    /// Estimate CUDA core utilization
    pub fn estimate_cuda_core_utilization(
        threads_per_block: u32,
        blocks_per_grid: u32,
        occupancy: f64,
    ) -> f64 {
        let total_threads = threads_per_block * blocks_per_grid;
        let rtx_5070_max_threads = 6144; // CUDA cores

        let utilization = (total_threads.min(rtx_5070_max_threads) as f64
            / rtx_5070_max_threads as f64)
            * occupancy;
        utilization * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_kernel_config() {
        let (block_size, grid_size) = Rtx5070Optimizer::get_optimal_kernel_config(1000);
        assert_eq!(block_size, 512);
        assert!(grid_size > 0);
    }

    #[test]
    fn test_bandwidth_calculation() {
        let utilization = Rtx5070Optimizer::calculate_memory_bandwidth_utilization(
            1024 * 1024 * 1024, // 1GB
            1000.0,             // 1ms
        );
        assert!(utilization > 0.0 && utilization <= 100.0);
    }
}
