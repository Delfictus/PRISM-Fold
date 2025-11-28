use super::WorldRecordConfig;
use crate::errors::PRCTError;
use std::fs;
use std::path::Path;

impl WorldRecordConfig {
    /// Load WorldRecordConfig from JSON or TOML file
    ///
    /// Automatically detects format based on file extension (.json or .toml).
    /// Validates all configuration including geodesic settings after parsing.
    ///
    /// # Arguments
    /// * `path` - Path to configuration file
    ///
    /// # Returns
    /// * `Ok(WorldRecordConfig)` - Validated configuration
    /// * `Err(PRCTError)` - Parse or validation error
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, PRCTError> {
        let path = path.as_ref();
        let data = fs::read_to_string(path)
            .map_err(|e| PRCTError::ColoringFailed(format!("read error: {e}")))?;

        let cfg = match path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
        {
            "toml" => toml::from_str::<WorldRecordConfig>(&data)
                .map_err(|e| PRCTError::ColoringFailed(format!("TOML parse error: {e}")))?,
            "json" => serde_json::from_str::<WorldRecordConfig>(&data)
                .map_err(|e| PRCTError::ColoringFailed(format!("JSON parse error: {e}")))?,
            _ => serde_json::from_str::<WorldRecordConfig>(&data)
                .or_else(|_| toml::from_str::<WorldRecordConfig>(&data))
                .map_err(|e| {
                    PRCTError::ColoringFailed(format!("Parse error (tried JSON then TOML): {e}"))
                })?,
        };

        cfg.validate()?;
        Ok(cfg)
    }
}
