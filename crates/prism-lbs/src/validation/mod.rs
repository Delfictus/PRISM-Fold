//! Validation, Metrics, and Explainability Module
//!
//! Comprehensive validation toolkit for PRISM-LBS:
//!
//! ## DCC/DCA Metrics
//! Industry-standard binding site prediction metrics:
//! - DCC (Distance to Closest Contact): Min distance from pocket to ligand
//! - DCA (Distance to Center of Active site): Centroid-to-centroid distance
//! - Top-N Success Rates: Success with best N pockets
//!
//! ## Ligand Parsing
//! Multi-format ligand coordinate extraction:
//! - PDB HETATM records
//! - SDF/MOL files
//! - MOL2 (Tripos format)
//! - XYZ coordinate files
//!
//! ## Benchmark Runner
//! Automated validation against standard datasets:
//! - PDBBind refined set
//! - scPDB druggable sites
//! - Custom benchmark sets
//!
//! ## Explainability
//! Interpretable AI for drug discovery:
//! - Per-residue contribution scores
//! - Druggability factor decomposition
//! - Confidence breakdown
//! - Human-readable reasoning
//!
//! ## Docking Integration
//! Seamless docking workflow support:
//! - AutoDock Vina box generation
//! - Pharmacophore feature extraction
//! - PyMOL visualization scripts

pub mod metrics;
pub mod ligand_parser;
pub mod benchmark;
pub mod explainability;
pub mod docking;

// Re-export key types
pub use metrics::{
    BenchmarkCase, BenchmarkSummary, CaseResult, TopNMetrics, ValidationMetrics,
    DEFAULT_SUCCESS_THRESHOLD,
};

pub use ligand_parser::{Ligand, LigandAtom, LigandParser};

pub use benchmark::{
    BenchmarkComparison, BenchmarkReport, CaseEvaluation, PDBBindBenchmark,
    validate_single,
};

pub use explainability::{
    AssessmentStatus, ConfidenceBreakdown, DetectionSignal, DruggabilityClass,
    DruggabilityFactors, ExplainabilityEngine, FactorAssessment, PocketExplanation,
    ResidueContribution, ResidueFactors, ResidueRole,
};

pub use docking::{
    DockingSite, DockingSiteGenerator, PharmacophoreFeature, PharmacophoreModel,
    PharmacophoreType, VinaBox,
};
