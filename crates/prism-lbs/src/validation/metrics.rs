//! Benchmark metrics utilities

use crate::pocket::Pocket;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct ValidationMetrics {
    pub center_distance: f64,
    pub ligand_coverage: f64,
    pub pocket_precision: f64,
    pub success_rate: f64,
}

impl ValidationMetrics {
    pub fn compute(pockets: &[Pocket], ligand: &[[f64; 3]], threshold: f64) -> Self {
        let best = pockets.iter().min_by(|a, b| {
            let da = distance(&a.centroid, ligand);
            let db = distance(&b.centroid, ligand);
            da.partial_cmp(&db).unwrap()
        });

        match best {
            Some(p) => {
                let dist = distance(&p.centroid, ligand);
                Self {
                    center_distance: dist,
                    ligand_coverage: coverage(p, ligand, threshold),
                    pocket_precision: precision(p, ligand, threshold),
                    success_rate: if dist <= threshold { 1.0 } else { 0.0 },
                }
            }
            None => Self::default(),
        }
    }
}

/// Benchmark case describing a structure and ligand coordinates
#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    pub name: String,
    pub ligand_coords: Vec<[f64; 3]>,
    pub threshold: f64,
}

impl BenchmarkCase {
    pub fn from_xyz(name: impl Into<String>, path: &Path, threshold: f64) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut coords = Vec::new();
        for line in content.lines() {
            let cols: Vec<_> = line
                .split_whitespace()
                .filter_map(|c| c.parse::<f64>().ok())
                .collect();
            if cols.len() == 3 {
                coords.push([cols[0], cols[1], cols[2]]);
            }
        }
        Ok(Self {
            name: name.into(),
            ligand_coords: coords,
            threshold,
        })
    }

    /// Load all `.xyz` files in a directory into benchmark cases with a uniform threshold.
    pub fn load_dir(dir: &Path, threshold: f64) -> std::io::Result<Vec<Self>> {
        let mut cases = Vec::new();
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    let path = entry.path();
                    if path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map_or(false, |ext| ext.eq_ignore_ascii_case("xyz"))
                    {
                        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("case");
                        cases.push(Self::from_xyz(name.to_string(), &path, threshold)?);
                    }
                }
            }
        }
        Ok(cases)
    }
}

/// Aggregate metrics across multiple benchmark cases
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct BenchmarkSummary {
    pub cases: usize,
    pub success_rate: f64,
    pub mean_center_distance: f64,
    pub mean_coverage: f64,
    pub mean_precision: f64,
}

impl BenchmarkSummary {
    pub fn evaluate(pockets: &[Pocket], cases: &[BenchmarkCase]) -> Self {
        if cases.is_empty() {
            return Self::default();
        }
        let mut success = 0.0;
        let mut d_sum = 0.0;
        let mut cov_sum = 0.0;
        let mut prec_sum = 0.0;
        for case in cases {
            let metrics = ValidationMetrics::compute(pockets, &case.ligand_coords, case.threshold);
            success += metrics.success_rate;
            d_sum += metrics.center_distance;
            cov_sum += metrics.ligand_coverage;
            prec_sum += metrics.pocket_precision;
        }
        let n = cases.len() as f64;
        Self {
            cases: cases.len(),
            success_rate: success / n,
            mean_center_distance: d_sum / n,
            mean_coverage: cov_sum / n,
            mean_precision: prec_sum / n,
        }
    }
}

fn distance(point: &[f64; 3], ligand: &[[f64; 3]]) -> f64 {
    ligand
        .iter()
        .map(|l| {
            let dx = point[0] - l[0];
            let dy = point[1] - l[1];
            let dz = point[2] - l[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn coverage(pocket: &Pocket, ligand: &[[f64; 3]], threshold: f64) -> f64 {
    if ligand.is_empty() {
        return 0.0;
    }
    let hits = ligand
        .iter()
        .filter(|l| {
            let dx = pocket.centroid[0] - l[0];
            let dy = pocket.centroid[1] - l[1];
            let dz = pocket.centroid[2] - l[2];
            (dx * dx + dy * dy + dz * dz).sqrt() <= threshold
        })
        .count();
    hits as f64 / ligand.len() as f64
}

fn precision(pocket: &Pocket, ligand: &[[f64; 3]], threshold: f64) -> f64 {
    if pocket.atom_indices.is_empty() {
        return 0.0;
    }
    let hits = pocket
        .atom_indices
        .iter()
        .filter(|_| {
            let dx = pocket.centroid[0] - ligand[0][0];
            let dy = pocket.centroid[1] - ligand[0][1];
            let dz = pocket.centroid[2] - ligand[0][2];
            (dx * dx + dy * dy + dz * dz).sqrt() <= threshold
        })
        .count();
    hits as f64 / pocket.atom_indices.len() as f64
}
