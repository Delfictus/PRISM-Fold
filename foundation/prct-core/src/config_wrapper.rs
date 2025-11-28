// Configuration Wrapper - Bridges existing configs with registry
// This allows runtime parameter injection without modifying source

use crate::world_record_pipeline::{WorldRecordConfig, GpuConfig, ThermoConfig, QuantumConfig};
use shared_types::config_registry::{CONFIG_REGISTRY, ParameterMetadata};
use serde_json::Value;
use std::sync::Arc;

/// Wrapper that intercepts config access and uses registry values
pub struct ConfigWrapper {
    base_config: WorldRecordConfig,
    prefix: String,
}

impl ConfigWrapper {
    pub fn new(mut config: WorldRecordConfig) -> Self {
        // First, register all parameters from the config
        Self::register_config(&config);

        // Then update config from registry (if values were pre-loaded)
        Self::update_config(&mut config);

        Self {
            base_config: config,
            prefix: String::new(),
        }
    }

    /// Register all fields from config structs
    fn register_config(config: &WorldRecordConfig) {
        // Top-level parameters
        CONFIG_REGISTRY.register_parameter(ParameterMetadata {
            name: "target_chromatic".to_string(),
            path: "target_chromatic".to_string(),
            value_type: "usize".to_string(),
            default: serde_json::to_value(config.target_chromatic).unwrap(),
            current: serde_json::to_value(config.target_chromatic).unwrap(),
            min: Some(Value::from(1)),
            max: Some(Value::from(10000)),
            description: "Target chromatic number for coloring".to_string(),
            category: "goals".to_string(),
            affects_gpu: false,
            requires_restart: false,
            access_count: 0,
        });

        CONFIG_REGISTRY.register_parameter(ParameterMetadata {
            name: "max_runtime_hours".to_string(),
            path: "max_runtime_hours".to_string(),
            value_type: "f64".to_string(),
            default: serde_json::to_value(config.max_runtime_hours).unwrap(),
            current: serde_json::to_value(config.max_runtime_hours).unwrap(),
            min: Some(Value::from(0.1)),
            max: Some(Value::from(168.0)), // 1 week
            description: "Maximum runtime in hours".to_string(),
            category: "limits".to_string(),
            affects_gpu: false,
            requires_restart: false,
            access_count: 0,
        });

        CONFIG_REGISTRY.register_parameter(ParameterMetadata {
            name: "deterministic".to_string(),
            path: "deterministic".to_string(),
            value_type: "bool".to_string(),
            default: serde_json::to_value(config.deterministic).unwrap(),
            current: serde_json::to_value(config.deterministic).unwrap(),
            min: None,
            max: None,
            description: "Use deterministic random seed".to_string(),
            category: "reproducibility".to_string(),
            affects_gpu: false,
            requires_restart: true,
            access_count: 0,
        });

        // GPU parameters
        Self::register_gpu_config(&config.gpu);

        // Thermodynamic parameters
        Self::register_thermo_config(&config.thermo);

        // Quantum parameters
        Self::register_quantum_config(&config.quantum);

        // Memetic parameters
        Self::register_memetic_config(&config.memetic);

        // Phase toggles
        let phase_toggles = [
            ("use_reservoir_prediction", config.use_reservoir_prediction, "Enable neuromorphic reservoir computing"),
            ("use_active_inference", config.use_active_inference, "Enable active inference"),
            ("use_transfer_entropy", config.use_transfer_entropy, "Enable transfer entropy analysis"),
            ("use_thermodynamic_equilibration", config.use_thermodynamic_equilibration, "Enable thermodynamic equilibration"),
            ("use_quantum_classical_hybrid", config.use_quantum_classical_hybrid, "Enable quantum-classical hybrid"),
            ("use_multiscale_analysis", config.use_multiscale_analysis, "Enable multi-scale analysis"),
            ("use_ensemble_consensus", config.use_ensemble_consensus, "Enable ensemble consensus voting"),
            ("use_geodesic_features", config.use_geodesic_features, "Enable geodesic feature computation"),
        ];

        for (name, value, desc) in phase_toggles.iter() {
            CONFIG_REGISTRY.register_parameter(ParameterMetadata {
                name: name.to_string(),
                path: name.to_string(),
                value_type: "bool".to_string(),
                default: serde_json::to_value(value).unwrap(),
                current: serde_json::to_value(value).unwrap(),
                min: None,
                max: None,
                description: desc.to_string(),
                category: "phases".to_string(),
                affects_gpu: name.contains("reservoir") || name.contains("thermodynamic") || name.contains("quantum"),
                requires_restart: false,
                access_count: 0,
            });
        }
    }

    fn register_gpu_config(config: &GpuConfig) {
        let gpu_params = [
            ("gpu.device_id", config.device_id as i64, 0, 8, "CUDA device ID"),
            ("gpu.streams", config.streams as i64, 1, 32, "Number of CUDA streams"),
            ("gpu.batch_size", config.batch_size as i64, 32, 8192, "GPU batch size"),
        ];

        for (path, value, min, max, desc) in gpu_params.iter() {
            CONFIG_REGISTRY.register_parameter(ParameterMetadata {
                name: path.split('.').last().unwrap().to_string(),
                path: path.to_string(),
                value_type: "usize".to_string(),
                default: serde_json::to_value(value).unwrap(),
                current: serde_json::to_value(value).unwrap(),
                min: Some(Value::from(*min)),
                max: Some(Value::from(*max)),
                description: desc.to_string(),
                category: "gpu".to_string(),
                affects_gpu: true,
                requires_restart: true,
                access_count: 0,
            });
        }

        // GPU feature flags
        let gpu_flags = [
            ("gpu.enable_reservoir_gpu", config.enable_reservoir_gpu, "GPU acceleration for reservoir"),
            ("gpu.enable_te_gpu", config.enable_te_gpu, "GPU acceleration for transfer entropy"),
            ("gpu.enable_statmech_gpu", config.enable_statmech_gpu, "GPU acceleration for stat mech"),
            ("gpu.enable_thermo_gpu", config.enable_thermo_gpu, "GPU acceleration for thermodynamic"),
            ("gpu.enable_quantum_gpu", config.enable_quantum_gpu, "GPU acceleration for quantum"),
        ];

        for (path, value, desc) in gpu_flags.iter() {
            CONFIG_REGISTRY.register_parameter(ParameterMetadata {
                name: path.split('.').last().unwrap().to_string(),
                path: path.to_string(),
                value_type: "bool".to_string(),
                default: serde_json::to_value(value).unwrap(),
                current: serde_json::to_value(value).unwrap(),
                min: None,
                max: None,
                description: desc.to_string(),
                category: "gpu".to_string(),
                affects_gpu: true,
                requires_restart: false,
                access_count: 0,
            });
        }
    }

    fn register_thermo_config(config: &ThermoConfig) {
        CONFIG_REGISTRY.register_parameter(ParameterMetadata {
            name: "replicas".to_string(),
            path: "thermo.replicas".to_string(),
            value_type: "usize".to_string(),
            default: serde_json::to_value(config.replicas).unwrap(),
            current: serde_json::to_value(config.replicas).unwrap(),
            min: Some(Value::from(1)),
            max: Some(Value::from(56)), // VRAM limit
            description: "Number of parallel temperature replicas".to_string(),
            category: "thermo".to_string(),
            affects_gpu: true,
            requires_restart: false,
            access_count: 0,
        });

        CONFIG_REGISTRY.register_parameter(ParameterMetadata {
            name: "num_temps".to_string(),
            path: "thermo.num_temps".to_string(),
            value_type: "usize".to_string(),
            default: serde_json::to_value(config.num_temps).unwrap(),
            current: serde_json::to_value(config.num_temps).unwrap(),
            min: Some(Value::from(2)),
            max: Some(Value::from(56)), // VRAM limit
            description: "Number of temperature levels".to_string(),
            category: "thermo".to_string(),
            affects_gpu: true,
            requires_restart: false,