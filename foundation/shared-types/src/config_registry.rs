// Universal Configuration Registry
// This provides runtime access to ALL parameters with verification

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global configuration registry accessible from anywhere
pub static CONFIG_REGISTRY: Lazy<Arc<ConfigRegistry>> =
    Lazy::new(|| Arc::new(ConfigRegistry::new()));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterMetadata {
    pub name: String,
    pub path: String,       // e.g., "thermo.replicas"
    pub value_type: String, // "usize", "f64", "bool"
    pub default: Value,
    pub current: Value,
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub description: String,
    pub category: String, // "gpu", "thermo", "quantum", etc.
    pub affects_gpu: bool,
    pub requires_restart: bool,
    pub access_count: usize, // For verification
}

pub struct ConfigRegistry {
    parameters: RwLock<HashMap<String, ParameterMetadata>>,
    access_log: RwLock<Vec<AccessRecord>>,
    verification_mode: RwLock<bool>,
}

#[derive(Debug, Clone)]
pub struct AccessRecord {
    pub timestamp: std::time::Instant,
    pub parameter: String,
    pub module: String,
    pub operation: AccessOp,
}

#[derive(Debug, Clone)]
pub enum AccessOp {
    Read,
    Write(Value),
    Validate,
}

impl ConfigRegistry {
    pub fn new() -> Self {
        Self {
            parameters: RwLock::new(HashMap::new()),
            access_log: RwLock::new(Vec::new()),
            verification_mode: RwLock::new(false),
        }
    }

    /// Register a parameter (called during initialization)
    pub fn register_parameter(&self, meta: ParameterMetadata) {
        let mut params = self.parameters.write().unwrap();
        params.insert(meta.path.clone(), meta);
    }

    /// Get parameter value with tracking
    pub fn get<T: for<'de> Deserialize<'de>>(&self, path: &str, module: &str) -> Option<T> {
        let params = self.parameters.read().unwrap();

        // Log access for verification
        if *self.verification_mode.read().unwrap() {
            let mut log = self.access_log.write().unwrap();
            log.push(AccessRecord {
                timestamp: std::time::Instant::now(),
                parameter: path.to_string(),
                module: module.to_string(),
                operation: AccessOp::Read,
            });
        }

        // Increment access count
        if let Some(param) = params.get(path) {
            drop(params);
            let mut params = self.parameters.write().unwrap();
            if let Some(param) = params.get_mut(path) {
                param.access_count += 1;
            }
            drop(params);
            let params = self.parameters.read().unwrap();

            // Return current value
            params
                .get(path)
                .and_then(|p| serde_json::from_value(p.current.clone()).ok())
        } else {
            None
        }
    }

    /// Set parameter value with validation
    pub fn set(&self, path: &str, value: Value) -> Result<(), String> {
        let mut params = self.parameters.write().unwrap();

        if let Some(param) = params.get_mut(path) {
            // Validate type
            if !Self::validate_type(&value, &param.value_type) {
                return Err(format!(
                    "Type mismatch for {}: expected {}",
                    path, param.value_type
                ));
            }

            // Validate bounds
            if let Some(min) = &param.min {
                if !Self::validate_bound(&value, min, true) {
                    return Err(format!("{} below minimum: {} < {}", path, value, min));
                }
            }

            if let Some(max) = &param.max {
                if !Self::validate_bound(&value, max, false) {
                    return Err(format!("{} above maximum: {} > {}", path, value, max));
                }
            }

            param.current = value.clone();

            // Log write
            drop(params);
            if *self.verification_mode.read().unwrap() {
                let mut log = self.access_log.write().unwrap();
                log.push(AccessRecord {
                    timestamp: std::time::Instant::now(),
                    parameter: path.to_string(),
                    module: "cli".to_string(),
                    operation: AccessOp::Write(value),
                });
            }

            Ok(())
        } else {
            Err(format!("Unknown parameter: {}", path))
        }
    }

    /// Load from TOML file with layered merging
    pub fn load_toml(&self, content: &str) -> Result<(), String> {
        let toml_value: toml::Value =
            toml::from_str(content).map_err(|e| format!("TOML parse error: {}", e))?;

        self.apply_toml_recursive(&toml_value, "")
    }

    fn apply_toml_recursive(&self, value: &toml::Value, prefix: &str) -> Result<(), String> {
        match value {
            toml::Value::Table(table) => {
                for (key, val) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", prefix, key)
                    };

                    if val.is_table() {
                        self.apply_toml_recursive(val, &path)?;
                    } else {
                        // Convert TOML to JSON value
                        let json_val = Self::toml_to_json(val);
                        self.set(&path, json_val)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Generate verification report
    pub fn generate_verification_report(&self) -> VerificationReport {
        let params = self.parameters.read().unwrap();
        let log = self.access_log.read().unwrap();

        let mut unused_params = Vec::new();
        let mut frequently_used = Vec::new();
        let mut modified_params = Vec::new();

        for (path, param) in params.iter() {
            if param.access_count == 0 {
                unused_params.push(path.clone());
            } else if param.access_count > 10 {
                frequently_used.push((path.clone(), param.access_count));
            }

            if param.current != param.default {
                modified_params.push(ModifiedParam {
                    path: path.clone(),
                    default: param.default.clone(),
                    current: param.current.clone(),
                });
            }
        }

        VerificationReport {
            total_parameters: params.len(),
            accessed_parameters: params.values().filter(|p| p.access_count > 0).count(),
            unused_parameters: unused_params,
            frequently_used,
            modified_parameters: modified_params,
            total_accesses: log.len(),
        }
    }

    fn validate_type(value: &Value, expected: &str) -> bool {
        match expected {
            "usize" | "u32" | "u64" => value.is_u64(),
            "isize" | "i32" | "i64" => value.is_i64() || value.is_u64(),
            "f32" | "f64" => value.is_f64() || value.is_i64() || value.is_u64(),
            "bool" => value.is_boolean(),
            "String" | "&str" => value.is_string(),
            _ => true,
        }
    }

    fn validate_bound(value: &Value, bound: &Value, is_min: bool) -> bool {
        if let (Some(v), Some(b)) = (value.as_f64(), bound.as_f64()) {
            if is_min {
                v >= b
            } else {
                v <= b
            }
        } else if let (Some(v), Some(b)) = (value.as_u64(), bound.as_u64()) {
            if is_min {
                v >= b
            } else {
                v <= b
            }
        } else if let (Some(v), Some(b)) = (value.as_i64(), bound.as_i64()) {
            if is_min {
                v >= b
            } else {
                v <= b
            }
        } else {
            true
        }
    }

    fn toml_to_json(toml_val: &toml::Value) -> Value {
        match toml_val {
            toml::Value::String(s) => Value::String(s.clone()),
            toml::Value::Integer(i) => Value::Number((*i).into()),
            toml::Value::Float(f) => {
                if let Some(n) = serde_json::Number::from_f64(*f) {
                    Value::Number(n)
                } else {
                    Value::Null
                }
            }
            toml::Value::Boolean(b) => Value::Bool(*b),
            toml::Value::Array(arr) => Value::Array(arr.iter().map(Self::toml_to_json).collect()),
            toml::Value::Table(table) => {
                let map: serde_json::Map<String, Value> = table
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::toml_to_json(v)))
                    .collect();
                Value::Object(map)
            }
            toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub total_parameters: usize,
    pub accessed_parameters: usize,
    pub unused_parameters: Vec<String>,
    pub frequently_used: Vec<(String, usize)>,
    pub modified_parameters: Vec<ModifiedParam>,
    pub total_accesses: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifiedParam {
    pub path: String,
    pub default: Value,
    pub current: Value,
}

/// Macro for easy parameter access with automatic registration
#[macro_export]
macro_rules! config_get {
    ($path:expr, $type:ty, $default:expr, $module:expr) => {{
        use $crate::config_registry::CONFIG_REGISTRY;

        CONFIG_REGISTRY
            .get::<$type>($path, $module)
            .unwrap_or_else(|| {
                // Auto-register on first access
                let meta = $crate::config_registry::ParameterMetadata {
                    name: $path.split('.').last().unwrap_or($path).to_string(),
                    path: $path.to_string(),
                    value_type: stringify!($type).to_string(),
                    default: serde_json::to_value($default).unwrap(),
                    current: serde_json::to_value($default).unwrap(),
                    min: None,
                    max: None,
                    description: format!("Auto-discovered parameter {}", $path),
                    category: $module.to_string(),
                    affects_gpu: $path.contains("gpu"),
                    requires_restart: false,
                    access_count: 1,
                };
                CONFIG_REGISTRY.register_parameter(meta);
                $default
            })
    }};
}

/// Helper to inject registry into existing config structs
pub trait ConfigRegistryInjector {
    fn inject_to_registry(&self, prefix: &str);
    fn update_from_registry(&mut self, prefix: &str);
}
