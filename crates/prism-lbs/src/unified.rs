//! Unified pocket detector combining geometric and soft-spot detection
//!
//! This module orchestrates both detection methods and produces a single,
//! merged output with detection type annotations.

use crate::graph::ProteinGraph;
use crate::pocket::{Pocket, PocketDetector, PocketDetectorConfig};
use crate::softspot::{
    CrypticCandidate, CrypticConfidence, EnhancedSoftSpotConfig, EnhancedSoftSpotDetector,
    SoftSpotDetector,
};
use crate::structure::Atom;
use crate::LbsError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for the unified detector
#[derive(Debug, Clone)]
pub struct UnifiedDetectorConfig {
    /// Enable geometric pocket detection
    pub enable_geometric: bool,

    /// Enable soft-spot (cryptic) detection
    pub enable_softspot: bool,

    /// Use enhanced multi-signal detection (NMA, Contact Order, Conservation, Probes)
    /// This is ADDITIVE - it can only improve detection, never reduce scores
    pub use_enhanced_detection: bool,

    /// Maximum pockets to return
    pub max_pockets: usize,

    /// Overlap threshold for consensus detection (0.0-1.0)
    /// Pockets with overlap > this are merged as consensus
    pub consensus_overlap_threshold: f64,
}

impl Default for UnifiedDetectorConfig {
    fn default() -> Self {
        Self {
            enable_geometric: true,
            enable_softspot: true,
            use_enhanced_detection: true, // Enhanced detection ON by default
            max_pockets: 20,
            consensus_overlap_threshold: 0.3,
        }
    }
}

/// Type of detection method that identified the pocket
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DetectionType {
    /// Detected by geometric cavity analysis (visible pocket)
    Geometric,
    /// Detected by soft-spot analysis (cryptic site)
    Cryptic,
    /// Detected by both methods (high confidence)
    Consensus,
}

/// Confidence level for unified detection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// High confidence (consensus or strong single-method signal)
    High,
    /// Medium confidence
    Medium,
    /// Low confidence
    Low,
}

/// Evidence supporting the detection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Evidence {
    /// Geometric druggability score (if detected geometrically)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometric_score: Option<f64>,

    /// Flexibility score from soft-spot analysis
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flexibility_score: Option<f64>,

    /// Packing deficit from soft-spot analysis
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packing_deficit: Option<f64>,

    /// Hydrophobicity score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hydrophobic_score: Option<f64>,

    // Enhanced detection signals (v2)
    /// NMA mobility score (induced-fit susceptibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nma_mobility: Option<f64>,

    /// Contact order flexibility (conformational change potential)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_order: Option<f64>,

    /// Evolutionary conservation score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conservation: Option<f64>,

    /// Probe clustering score (druggable hot spots)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_score: Option<f64>,

    /// Which detectors identified this pocket
    pub detected_by: Vec<String>,
}

/// A unified pocket combining geometric and cryptic detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedPocket {
    /// Pocket ID (1-indexed)
    pub id: usize,

    /// Residue sequence numbers (PDB RESSEQ)
    pub residue_indices: Vec<i32>,

    /// Centroid [x, y, z] in Angstroms
    pub centroid: [f64; 3],

    /// Volume in cubic Angstroms
    pub volume: f64,

    /// Druggability score (0.0-1.0)
    pub druggability: f64,

    /// How the pocket was detected
    pub detection_type: DetectionType,

    /// Confidence level
    pub confidence: Confidence,

    /// Evidence supporting the detection
    pub evidence: Evidence,
}

/// Summary statistics for unified detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSummary {
    /// Total pockets detected
    pub total_pockets: usize,

    /// Pockets from geometric detection only
    pub geometric: usize,

    /// Pockets from cryptic detection only
    pub cryptic: usize,

    /// Pockets detected by both methods
    pub consensus: usize,
}

/// Complete output from unified detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedOutput {
    /// Structure name/identifier
    pub structure: String,

    /// Detected pockets
    pub pockets: Vec<UnifiedPocket>,

    /// Summary statistics
    pub summary: UnifiedSummary,
}

/// Unified detector combining geometric and soft-spot detection
pub struct UnifiedDetector {
    pub config: UnifiedDetectorConfig,
    geometric_config: PocketDetectorConfig,
    softspot: SoftSpotDetector,
    enhanced_softspot: EnhancedSoftSpotDetector,
}

impl Default for UnifiedDetector {
    fn default() -> Self {
        Self {
            config: UnifiedDetectorConfig::default(),
            geometric_config: PocketDetectorConfig::default(),
            softspot: SoftSpotDetector::new(),
            enhanced_softspot: EnhancedSoftSpotDetector::new(),
        }
    }
}

impl UnifiedDetector {
    /// Create a new unified detector with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a unified detector with custom configuration
    pub fn with_config(config: UnifiedDetectorConfig) -> Self {
        Self {
            config,
            geometric_config: PocketDetectorConfig::default(),
            softspot: SoftSpotDetector::new(),
            enhanced_softspot: EnhancedSoftSpotDetector::new(),
        }
    }

    /// Create a unified detector with custom enhanced configuration
    pub fn with_enhanced_config(
        config: UnifiedDetectorConfig,
        enhanced_config: EnhancedSoftSpotConfig,
    ) -> Self {
        Self {
            config,
            geometric_config: PocketDetectorConfig::default(),
            softspot: SoftSpotDetector::new(),
            enhanced_softspot: EnhancedSoftSpotDetector::with_config(enhanced_config),
        }
    }

    /// Detect pockets using both methods
    ///
    /// # Arguments
    /// * `graph` - Protein graph for geometric detection
    /// * `structure_name` - Name to include in output
    ///
    /// # Returns
    /// Unified output combining all detected pockets
    pub fn detect(&self, graph: &ProteinGraph, structure_name: &str) -> Result<UnifiedOutput, LbsError> {
        log::info!(
            "[UNIFIED] Starting unified detection for {} (enhanced={})",
            structure_name,
            self.config.use_enhanced_detection
        );

        let mut geometric_pockets = Vec::new();
        let mut cryptic_candidates = Vec::new();

        // Run geometric detection
        if self.config.enable_geometric {
            let detector = PocketDetector {
                config: self.geometric_config.clone(),
            };
            geometric_pockets = detector.detect(graph)?;
            log::info!(
                "[UNIFIED] Geometric detection: {} pockets",
                geometric_pockets.len()
            );
        }

        // Run soft-spot detection (classic or enhanced)
        if self.config.enable_softspot {
            if self.config.use_enhanced_detection {
                // Enhanced multi-signal detection
                cryptic_candidates = self.enhanced_softspot.detect(&graph.structure_ref.atoms);
                log::info!(
                    "[UNIFIED] Enhanced soft-spot detection: {} candidates (NMA+CO+Cons+Probe)",
                    cryptic_candidates.len()
                );
            } else {
                // Classic B-factor-only detection
                cryptic_candidates = self.softspot.detect(&graph.structure_ref.atoms);
                log::info!(
                    "[UNIFIED] Classic soft-spot detection: {} candidates",
                    cryptic_candidates.len()
                );
            }
        }

        // Merge results
        let unified = self.merge_results(&geometric_pockets, &cryptic_candidates, graph);

        // Build summary
        let summary = UnifiedSummary {
            total_pockets: unified.len(),
            geometric: unified
                .iter()
                .filter(|p| p.detection_type == DetectionType::Geometric)
                .count(),
            cryptic: unified
                .iter()
                .filter(|p| p.detection_type == DetectionType::Cryptic)
                .count(),
            consensus: unified
                .iter()
                .filter(|p| p.detection_type == DetectionType::Consensus)
                .count(),
        };

        log::info!(
            "[UNIFIED] Final: {} total ({} geometric, {} cryptic, {} consensus)",
            summary.total_pockets,
            summary.geometric,
            summary.cryptic,
            summary.consensus
        );

        Ok(UnifiedOutput {
            structure: structure_name.to_string(),
            pockets: unified,
            summary,
        })
    }

    /// Detect pockets directly from atoms (convenience method)
    ///
    /// Note: This only runs soft-spot detection since geometric detection
    /// requires a full ProteinGraph.
    pub fn detect_from_atoms(&self, atoms: &[Atom], structure_name: &str) -> UnifiedOutput {
        log::info!(
            "[UNIFIED] Soft-spot only detection for {} ({} atoms, enhanced={})",
            structure_name,
            atoms.len(),
            self.config.use_enhanced_detection
        );

        let cryptic_candidates = if self.config.enable_softspot {
            if self.config.use_enhanced_detection {
                self.enhanced_softspot.detect(atoms)
            } else {
                self.softspot.detect(atoms)
            }
        } else {
            Vec::new()
        };

        // Convert cryptic candidates to unified pockets
        let unified: Vec<UnifiedPocket> = cryptic_candidates
            .into_iter()
            .enumerate()
            .map(|(i, cc)| self.cryptic_to_unified(i + 1, cc))
            .collect();

        let summary = UnifiedSummary {
            total_pockets: unified.len(),
            geometric: 0,
            cryptic: unified.len(),
            consensus: 0,
        };

        UnifiedOutput {
            structure: structure_name.to_string(),
            pockets: unified,
            summary,
        }
    }

    /// Merge geometric pockets and cryptic candidates
    fn merge_results(
        &self,
        geometric: &[Pocket],
        cryptic: &[CrypticCandidate],
        graph: &ProteinGraph,
    ) -> Vec<UnifiedPocket> {
        let mut unified = Vec::new();
        let mut cryptic_used = vec![false; cryptic.len()];

        // Process geometric pockets first
        for gp in geometric {
            // Convert geometric residue indices to RESSEQ
            let geo_residues: Vec<i32> = gp
                .residue_indices
                .iter()
                .filter_map(|&idx| {
                    graph.structure_ref.residues.get(idx).map(|r| r.seq_number)
                })
                .collect();

            let mut pocket = UnifiedPocket {
                id: unified.len() + 1,
                residue_indices: geo_residues.clone(),
                centroid: gp.centroid,
                volume: gp.volume,
                druggability: gp.druggability_score.total,
                detection_type: DetectionType::Geometric,
                confidence: Confidence::High,
                evidence: Evidence {
                    geometric_score: Some(gp.druggability_score.total),
                    flexibility_score: None,
                    packing_deficit: None,
                    hydrophobic_score: None,
                    nma_mobility: None,
                    contact_order: None,
                    conservation: None,
                    probe_score: None,
                    detected_by: vec!["geometric".into()],
                },
            };

            // Check for cryptic overlap
            for (i, cc) in cryptic.iter().enumerate() {
                if cryptic_used[i] {
                    continue;
                }

                let overlap = Self::residue_overlap(&geo_residues, &cc.residue_indices);
                if overlap > self.config.consensus_overlap_threshold {
                    // Mark as consensus
                    pocket.detection_type = DetectionType::Consensus;
                    pocket.evidence.flexibility_score = Some(cc.flexibility_score);
                    pocket.evidence.packing_deficit = Some(cc.packing_deficit);
                    pocket.evidence.hydrophobic_score = Some(cc.hydrophobic_score);
                    pocket.evidence.detected_by.push("softspot".into());
                    cryptic_used[i] = true;
                }
            }

            unified.push(pocket);
        }

        // Add cryptic-only candidates
        for (i, cc) in cryptic.iter().enumerate() {
            if cryptic_used[i] {
                continue;
            }

            unified.push(self.cryptic_to_unified(unified.len() + 1, cc.clone()));
        }

        // Sort by ranking score (highest first)
        unified.sort_by(|a, b| {
            Self::rank_score(b)
                .partial_cmp(&Self::rank_score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-assign IDs and truncate
        for (i, pocket) in unified.iter_mut().enumerate() {
            pocket.id = i + 1;
        }

        unified.truncate(self.config.max_pockets);
        unified
    }

    /// Convert a cryptic candidate to unified pocket format
    fn cryptic_to_unified(&self, id: usize, cc: CrypticCandidate) -> UnifiedPocket {
        let confidence = match cc.confidence {
            CrypticConfidence::High => Confidence::Medium, // Cryptic-only never "High"
            CrypticConfidence::Medium => Confidence::Medium,
            CrypticConfidence::Low => Confidence::Low,
        };

        UnifiedPocket {
            id,
            residue_indices: cc.residue_indices,
            centroid: cc.centroid,
            volume: cc.estimated_volume,
            druggability: cc.predicted_druggability,
            detection_type: DetectionType::Cryptic,
            confidence,
            evidence: Evidence {
                geometric_score: None,
                flexibility_score: Some(cc.flexibility_score),
                packing_deficit: Some(cc.packing_deficit),
                hydrophobic_score: Some(cc.hydrophobic_score),
                nma_mobility: None,     // TODO: Extract from enhanced rationale
                contact_order: None,    // TODO: Extract from enhanced rationale
                conservation: None,     // TODO: Extract from enhanced rationale
                probe_score: None,      // TODO: Extract from enhanced rationale
                detected_by: vec!["softspot".into()],
            },
        }
    }

    /// Calculate residue overlap between two pocket definitions
    fn residue_overlap(a: &[i32], b: &[i32]) -> f64 {
        let set_a: HashSet<_> = a.iter().collect();
        let set_b: HashSet<_> = b.iter().collect();
        let intersection = set_a.intersection(&set_b).count();
        let smaller = a.len().min(b.len());
        if smaller == 0 {
            0.0
        } else {
            intersection as f64 / smaller as f64
        }
    }

    /// Calculate ranking score for sorting
    ///
    /// Consensus pockets rank highest, then geometric, then cryptic
    fn rank_score(p: &UnifiedPocket) -> f64 {
        let base = p.druggability;
        match p.detection_type {
            DetectionType::Consensus => base * 1.2,
            DetectionType::Geometric => base * 1.0,
            DetectionType::Cryptic => base * 0.85,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_overlap() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![4, 5, 6, 7];
        let overlap = UnifiedDetector::residue_overlap(&a, &b);
        assert!((overlap - 0.5).abs() < 0.01); // 2 common out of 4 (smaller set)

        let c = vec![10, 20, 30];
        let overlap_none = UnifiedDetector::residue_overlap(&a, &c);
        assert_eq!(overlap_none, 0.0);
    }

    #[test]
    fn test_rank_score() {
        let consensus = UnifiedPocket {
            id: 1,
            residue_indices: vec![1],
            centroid: [0.0; 3],
            volume: 100.0,
            druggability: 0.5,
            detection_type: DetectionType::Consensus,
            confidence: Confidence::High,
            evidence: Evidence::default(),
        };

        let geometric = UnifiedPocket {
            detection_type: DetectionType::Geometric,
            ..consensus.clone()
        };

        let cryptic = UnifiedPocket {
            detection_type: DetectionType::Cryptic,
            ..consensus.clone()
        };

        assert!(UnifiedDetector::rank_score(&consensus) > UnifiedDetector::rank_score(&geometric));
        assert!(UnifiedDetector::rank_score(&geometric) > UnifiedDetector::rank_score(&cryptic));
    }
}
