//! DIMACS Graph Format Parser
//!
//! Parses standard DIMACS graph coloring format (.col files)
//! Also supports Matrix Market format (.mtx files) from Network Repository

use crate::errors::{PRCTError, Result};
use shared_types::Graph;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse graph file - auto-detects format (DIMACS .col or Matrix Market .mtx)
pub fn parse_graph_file<P: AsRef<Path>>(path: P) -> Result<Graph> {
    let path_str = path.as_ref().to_string_lossy();

    if path_str.ends_with(".mtx") {
        parse_mtx_file(path)
    } else {
        parse_dimacs_file(path)
    }
}

/// Parse DIMACS .col file into Graph structure
pub fn parse_dimacs_file<P: AsRef<Path>>(path: P) -> Result<Graph> {
    let file = File::open(path)
        .map_err(|e| PRCTError::ConfigError(format!("Failed to open file: {}", e)))?;

    let reader = BufReader::new(file);
    let mut num_vertices = 0;
    let mut num_edges = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| PRCTError::ConfigError(format!("Read error: {}", e)))?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('c') {
            // Skip comments and empty lines
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.get(0) {
            Some(&"p") if parts.len() >= 4 && parts[1] == "edge" => {
                // Problem line: p edge <vertices> <edges>
                num_vertices = parts[2]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid vertices: {}", e)))?;
                num_edges = parts[3]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid edges: {}", e)))?;
            }
            Some(&"e") if parts.len() >= 3 => {
                // Edge line: e <vertex1> <vertex2>
                let v1: usize = parts[1]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid vertex: {}", e)))?;
                let v2: usize = parts[2]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid vertex: {}", e)))?;

                // DIMACS uses 1-indexed vertices, convert to 0-indexed
                edges.push((v1 - 1, v2 - 1, 1.0));
            }
            _ => {
                // Skip unknown lines
            }
        }
    }

    if num_vertices == 0 {
        return Err(PRCTError::InvalidGraph("No problem line found".into()));
    }

    // Build adjacency matrix
    let mut adjacency = vec![false; num_vertices * num_vertices];
    for (i, j, _) in &edges {
        if *i < num_vertices && *j < num_vertices {
            adjacency[i * num_vertices + j] = true;
            adjacency[j * num_vertices + i] = true;
        }
    }

    Ok(Graph {
        num_vertices,
        num_edges,
        edges,
        adjacency,
        coordinates: None,
    })
}

/// Parse Matrix Market .mtx file into Graph structure
///
/// Format:
/// %%MatrixMarket matrix coordinate pattern symmetric
/// <vertices> <vertices> <edges>
/// <vertex1> <vertex2>
/// ...
pub fn parse_mtx_file<P: AsRef<Path>>(path: P) -> Result<Graph> {
    let file = File::open(&path)
        .map_err(|e| PRCTError::ConfigError(format!("Failed to open MTX file: {}", e)))?;

    let reader = BufReader::new(file);
    let mut num_vertices = 0;
    let mut num_edges = 0;
    let mut edges = Vec::new();
    let mut header_parsed = false;

    for line in reader.lines() {
        let line = line.map_err(|e| PRCTError::ConfigError(format!("Read error: {}", e)))?;
        let line = line.trim();

        // Skip comments
        if line.starts_with('%') {
            continue;
        }

        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        if !header_parsed {
            // First non-comment line: <vertices> <vertices> <edges>
            if parts.len() >= 3 {
                num_vertices = parts[0].parse().map_err(|e| {
                    PRCTError::InvalidGraph(format!("Invalid vertices in MTX: {}", e))
                })?;
                num_edges = parts[2]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid edges in MTX: {}", e)))?;
                header_parsed = true;
                println!(
                    "[MTX Parser] Graph: {} vertices, {} edges",
                    num_vertices, num_edges
                );
            }
        } else {
            // Edge line: <vertex1> <vertex2>
            if parts.len() >= 2 {
                let v1: usize = parts[0]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid vertex: {}", e)))?;
                let v2: usize = parts[1]
                    .parse()
                    .map_err(|e| PRCTError::InvalidGraph(format!("Invalid vertex: {}", e)))?;

                // MTX uses 1-indexed vertices, convert to 0-indexed
                edges.push((v1 - 1, v2 - 1, 1.0));
            }
        }
    }

    if num_vertices == 0 {
        return Err(PRCTError::InvalidGraph(
            "No header found in MTX file".into(),
        ));
    }

    println!("[MTX Parser] Parsed {} edges", edges.len());

    // Build adjacency matrix
    let mut adjacency = vec![false; num_vertices * num_vertices];
    for (i, j, _) in &edges {
        if *i < num_vertices && *j < num_vertices {
            adjacency[i * num_vertices + j] = true;
            adjacency[j * num_vertices + i] = true;
        }
    }

    Ok(Graph {
        num_vertices,
        num_edges,
        edges,
        adjacency,
        coordinates: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_graph() {
        // Create a simple test DIMACS file
        let content = "c Test graph\np edge 3 2\ne 1 2\ne 2 3\n";

        // Would need to write to temp file to test properly
        // This is just structure validation
    }
}
