//! Provenance metadata for PRISM-LBS outputs
//!
//! Automatically generates companion .METADATA files for every output file.
//! This cannot be disabled - every output file gets provenance tracking.

use chrono::Utc;
use sha2::{Digest, Sha384};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

/// Write provenance metadata file alongside an output file.
///
/// This function is called automatically after every output file write.
/// It creates a companion .METADATA file with full provenance information.
///
/// # Arguments
/// * `output_path` - Path to the output file that was just written
/// * `start_time` - Instant when processing started
/// * `structure_count` - Number of structures processed
pub fn write_provenance_metadata(
    output_path: &Path,
    start_time: Instant,
    structure_count: usize,
) -> std::io::Result<()> {
    let elapsed = start_time.elapsed().as_secs_f64();
    let per_structure_ms = if structure_count > 0 {
        elapsed * 1000.0 / structure_count as f64
    } else {
        0.0
    };

    // Get binary information
    let binary_path = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Get git information
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    let tag = Command::new("git")
        .args(["describe", "--tags", "--always"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    // Get GPU information
    let gpu = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    let driver = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    // Calculate binary SHA384
    let binary_sha = if let Ok(binary_bytes) = fs::read(&binary_path) {
        format!("{:x}", Sha384::digest(&binary_bytes))
    } else {
        "unable_to_read_binary".to_string()
    };

    // Calculate output file SHA384
    let output_sha = if let Ok(output_bytes) = fs::read(output_path) {
        format!("{:x}", Sha384::digest(&output_bytes))
    } else {
        "unable_to_read_output".to_string()
    };

    // Build metadata path: foo.json -> foo.json.METADATA
    let ext = output_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let metadata_filename = format!(
        "{}.METADATA",
        output_path.file_name().unwrap_or_default().to_string_lossy()
    );
    let metadata_path = output_path.with_file_name(metadata_filename);

    // Get command line
    let command_line: String = std::env::args().collect::<Vec<_>>().join(" ");

    // Check if pure-GPU mode was used
    let pure_gpu_mode = command_line.contains("--pure-gpu") || std::env::var("PRISM_PURE_GPU").is_ok();

    let content = format!(
        r#"PRISM-LBS Provenance — DO NOT SEPARATE
=================================================
Generated: {}
Command: {}
Binary: {}
Binary SHA384: {}
Commit: {}
Tag: {}
Pure-GPU Mode: {}
Total structures: {}
Total runtime (s): {:.3}
Per-structure (ms): {:.2}
GPU: {}
Driver: {}
Output file: {}
Output file SHA384: {}

This file and its companion output must never be separated.
Copyright © 2025 Delfictus I/O Inc.
"#,
        Utc::now().to_rfc3339(),
        command_line,
        binary_path,
        binary_sha,
        commit.trim(),
        tag.trim(),
        pure_gpu_mode,
        structure_count,
        elapsed,
        per_structure_ms,
        gpu.trim(),
        driver.trim(),
        output_path.display(),
        output_sha
    );

    fs::write(&metadata_path, content)?;
    log::debug!("Wrote provenance metadata to {:?}", metadata_path);
    Ok(())
}

/// Write batch provenance metadata for multiple structures
///
/// Creates a single metadata file for a batch run with aggregate statistics.
pub fn write_batch_provenance_metadata(
    output_dir: &Path,
    start_time: Instant,
    structure_count: usize,
    output_files: &[&Path],
) -> std::io::Result<()> {
    let elapsed = start_time.elapsed().as_secs_f64();
    let per_structure_ms = if structure_count > 0 {
        elapsed * 1000.0 / structure_count as f64
    } else {
        0.0
    };

    // Get binary information
    let binary_path = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Get git information
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    let tag = Command::new("git")
        .args(["describe", "--tags", "--always"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    // Get GPU information
    let gpu = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    let driver = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string());

    // Calculate binary SHA384
    let binary_sha = if let Ok(binary_bytes) = fs::read(&binary_path) {
        format!("{:x}", Sha384::digest(&binary_bytes))
    } else {
        "unable_to_read_binary".to_string()
    };

    // Get command line
    let command_line: String = std::env::args().collect::<Vec<_>>().join(" ");

    // Check if pure-GPU mode was used
    let pure_gpu_mode = command_line.contains("--pure-gpu") || std::env::var("PRISM_PURE_GPU").is_ok();

    // Build output file list with checksums
    let mut output_checksums = String::new();
    for path in output_files.iter().take(100) {
        // Limit to first 100 files
        if let Ok(bytes) = fs::read(path) {
            let sha = format!("{:x}", Sha384::digest(&bytes));
            output_checksums.push_str(&format!("  {} {}\n", sha, path.display()));
        }
    }
    if output_files.len() > 100 {
        output_checksums.push_str(&format!("  ... and {} more files\n", output_files.len() - 100));
    }

    let metadata_path = output_dir.join("BATCH_PROVENANCE.METADATA");

    let content = format!(
        r#"PRISM-LBS Batch Provenance — DO NOT SEPARATE
=================================================
Generated: {}
Command: {}
Binary: {}
Binary SHA384: {}
Commit: {}
Tag: {}
Pure-GPU Mode: {}
Total structures: {}
Total runtime (s): {:.3}
Per-structure (ms): {:.2}
GPU: {}
Driver: {}
Output directory: {}
Output files: {}

Output Checksums (SHA384):
{}
This file and the output directory must never be separated.
Copyright © 2025 Delfictus I/O Inc.
"#,
        Utc::now().to_rfc3339(),
        command_line,
        binary_path,
        binary_sha,
        commit.trim(),
        tag.trim(),
        pure_gpu_mode,
        structure_count,
        elapsed,
        per_structure_ms,
        gpu.trim(),
        driver.trim(),
        output_dir.display(),
        output_files.len(),
        output_checksums
    );

    fs::write(&metadata_path, content)?;
    log::info!("Wrote batch provenance metadata to {:?}", metadata_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use tempfile::tempdir;

    #[test]
    fn test_provenance_metadata_creation() {
        let dir = tempdir().unwrap();
        let output_file = dir.path().join("test_output.json");
        fs::write(&output_file, r#"{"test": true}"#).unwrap();

        let start = Instant::now();
        let result = write_provenance_metadata(&output_file, start, 1);
        assert!(result.is_ok());

        let metadata_path = dir.path().join("test_output.json.METADATA");
        assert!(metadata_path.exists());

        let content = fs::read_to_string(&metadata_path).unwrap();
        assert!(content.contains("PRISM-LBS Provenance"));
        assert!(content.contains("Total structures: 1"));
        assert!(content.contains("Output file SHA384:"));
    }

    #[test]
    fn test_batch_provenance_metadata() {
        let dir = tempdir().unwrap();
        let output1 = dir.path().join("out1.json");
        let output2 = dir.path().join("out2.json");
        fs::write(&output1, "test1").unwrap();
        fs::write(&output2, "test2").unwrap();

        let start = Instant::now();
        let files: Vec<&Path> = vec![output1.as_path(), output2.as_path()];
        let result = write_batch_provenance_metadata(dir.path(), start, 2, &files);
        assert!(result.is_ok());

        let metadata_path = dir.path().join("BATCH_PROVENANCE.METADATA");
        assert!(metadata_path.exists());

        let content = fs::read_to_string(&metadata_path).unwrap();
        assert!(content.contains("PRISM-LBS Batch Provenance"));
        assert!(content.contains("Total structures: 2"));
    }
}
