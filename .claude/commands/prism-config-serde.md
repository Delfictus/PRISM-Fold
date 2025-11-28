# PRISM Config Serde Migration Agent

You are a specialized agent for migrating PRISM's brittle TOML config parsing to serde-based deserialization with validation.

## Mission
Replace 85+ manual `config.` extraction patterns in `prism-cli/src/main.rs` with:
- Serde-derived config structs
- Automatic TOML deserialization
- Compile-time type safety
- Runtime validation via `validator` crate
- Default values for optional fields

## Current Problem
File: `crates/prism-cli/src/main.rs` (1672 lines, 85 config extractions)
```rust
// CURRENT: Fragile, verbose, no validation
if let Some(phase2_table) = toml_config.get("phase2") {
    if let Some(iterations) = phase2_table.get("iterations").and_then(|v| v.as_integer()) {
        phase2_config.iterations = iterations as usize;
    }
    if let Some(replicas) = phase2_table.get("replicas").and_then(|v| v.as_integer()) {
        phase2_config.replicas = replicas as usize;
    }
    // ... repeat 20+ times
}
```

## Solution: Serde + Validator

### 1. Add Dependencies
File: `crates/prism-core/Cargo.toml`
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
validator = { version = "0.18", features = ["derive"] }
```

### 2. Define Config Structs
File: `crates/prism-core/src/config/mod.rs`
```rust
use serde::{Deserialize, Serialize};
use validator::Validate;
use std::path::PathBuf;

/// Root configuration for PRISM pipeline
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct PrismConfig {
    #[serde(default)]
    pub general: GeneralConfig,

    #[serde(default)]
    pub gpu: GpuConfig,

    #[serde(default)]
    pub warmstart: WarmstartConfig,

    #[serde(default)]
    pub fluxnet: FluxNetConfig,

    #[validate(nested)]
    pub phase0_dendritic: Option<Phase0Config>,

    #[validate(nested)]
    pub phase1_active_inference: Option<Phase1Config>,

    #[validate(nested)]
    pub phase2: Option<Phase2Config>,

    #[validate(nested)]
    pub phase3_quantum: Option<Phase3Config>,

    #[validate(nested)]
    pub phase4_geodesic: Option<Phase4Config>,

    #[validate(nested)]
    pub phase5_mec: Option<Phase5Config>,

    #[validate(nested)]
    pub phase6_belief: Option<Phase6Config>,

    #[validate(nested)]
    pub phase7_whcr: Option<Phase7Config>,

    #[validate(nested)]
    pub lbs: Option<LbsConfig>,
}

/// General pipeline settings
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GeneralConfig {
    #[serde(default = "default_max_colors")]
    pub max_colors: usize,

    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    #[serde(default)]
    pub verbose: bool,
}

/// GPU configuration
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct GpuConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_ptx_dir")]
    pub ptx_dir: PathBuf,

    #[serde(default)]
    pub require_signed_ptx: bool,

    #[serde(default)]
    pub allow_nvrtc: bool,

    #[serde(default = "default_nvml_poll")]
    pub nvml_poll_interval_ms: u64,

    pub trusted_ptx_dir: Option<PathBuf>,
}

/// Phase 2 (Thermodynamic) configuration
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct Phase2Config {
    #[validate(range(min = 1, max = 1_000_000))]
    #[serde(default = "default_phase2_iterations")]
    pub iterations: usize,

    #[validate(range(min = 1, max = 10_000))]
    #[serde(default = "default_phase2_replicas")]
    pub replicas: usize,

    #[validate(range(min = 0.0, max = 1000.0))]
    #[serde(default = "default_temp_min")]
    pub temp_min: f32,

    #[validate(range(min = 0.0, max = 10000.0))]
    #[serde(default = "default_temp_max")]
    pub temp_max: f32,

    #[serde(default = "default_cooling_schedule")]
    pub cooling_schedule: CoolingSchedule,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoolingSchedule {
    #[default]
    Exponential,
    Linear,
    Logarithmic,
    Adaptive,
}

/// LBS-specific configuration
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct LbsConfig {
    #[validate(range(min = 1.0, max = 3.0))]
    #[serde(default = "default_probe_radius")]
    pub probe_radius: f32,

    #[validate(range(min = 1, max = 1000))]
    #[serde(default = "default_min_pocket_size")]
    pub min_pocket_size: usize,

    #[validate(custom(function = "validate_path_exists"))]
    pub gnn_model_path: Option<PathBuf>,

    pub gnn_weights_path: Option<PathBuf>,

    #[serde(default)]
    pub gpu_enabled: bool,
}

// Default value functions
fn default_max_colors() -> usize { 100 }
fn default_max_iterations() -> usize { 10000 }
fn default_true() -> bool { true }
fn default_ptx_dir() -> PathBuf { PathBuf::from("kernels/ptx") }
fn default_nvml_poll() -> u64 { 1000 }
fn default_phase2_iterations() -> usize { 10000 }
fn default_phase2_replicas() -> usize { 8 }
fn default_temp_min() -> f32 { 0.1 }
fn default_temp_max() -> f32 { 100.0 }
fn default_cooling_schedule() -> CoolingSchedule { CoolingSchedule::Exponential }
fn default_probe_radius() -> f32 { 1.4 }
fn default_min_pocket_size() -> usize { 10 }

// Custom validators
fn validate_path_exists(path: &Option<PathBuf>) -> Result<(), validator::ValidationError> {
    if let Some(p) = path {
        if !p.exists() {
            let mut err = validator::ValidationError::new("path_not_found");
            err.message = Some(format!("Path does not exist: {}", p.display()).into());
            return Err(err);
        }
    }
    Ok(())
}
```

### 3. Config Loading
File: `crates/prism-core/src/config/loader.rs`
```rust
use super::PrismConfig;
use anyhow::{Context, Result};
use std::path::Path;
use validator::Validate;

impl PrismConfig {
    /// Load configuration from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config: {}", path.as_ref().display()))?;

        Self::from_toml(&content)
    }

    /// Parse configuration from TOML string
    pub fn from_toml(content: &str) -> Result<Self> {
        let config: PrismConfig = toml::from_str(content)
            .context("Failed to parse TOML configuration")?;

        config.validate()
            .context("Configuration validation failed")?;

        Ok(config)
    }

    /// Merge with CLI overrides
    pub fn with_overrides(mut self, args: &crate::cli::Args) -> Self {
        if let Some(max_colors) = args.max_colors {
            self.general.max_colors = max_colors;
        }
        if args.verbose {
            self.general.verbose = true;
        }
        // ... other overrides
        self
    }
}
```

### 4. Update CLI
File: `crates/prism-cli/src/main.rs`
```rust
use prism_core::config::PrismConfig;

fn main() -> Result<()> {
    let args = Args::parse();

    // Load config with serde (replaces 85 manual extractions!)
    let config = if let Some(ref config_path) = args.config {
        PrismConfig::load(config_path)?
            .with_overrides(&args)
    } else {
        PrismConfig::default()
            .with_overrides(&args)
    };

    // Validation happens automatically during load
    log::info!("Configuration loaded: {:?}", config);

    // Use typed config
    run_pipeline(config)
}
```

## Files to Modify
1. `crates/prism-core/Cargo.toml` - Add serde, validator
2. `crates/prism-core/src/config/mod.rs` - New module
3. `crates/prism-core/src/config/loader.rs` - Loading logic
4. `crates/prism-core/src/lib.rs` - Export config module
5. `crates/prism-cli/Cargo.toml` - Add toml
6. `crates/prism-cli/src/main.rs` - Replace manual parsing

## Validation
```bash
# Create test config
cat > /tmp/test_config.toml << 'EOF'
[general]
max_colors = 50
verbose = true

[phase2]
iterations = 5000
replicas = 4
temp_min = 0.5
temp_max = 200.0

[lbs]
probe_radius = 1.4
min_pocket_size = 15
EOF

# Test loading
cargo test --package prism-core -- config
cargo run --release -- --config /tmp/test_config.toml --help
```

## Output Format
Report:
1. Config structs defined (count)
2. CLI lines reduced (from ~1672 to ~X)
3. Validation rules added
4. Test coverage for config loading
5. Example config files created
