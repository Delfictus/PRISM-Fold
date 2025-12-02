//! Output utilities

pub mod json_export;
pub mod pocket_writer;
pub mod provenance;
pub mod publication;
pub mod visualization;

pub use json_export::write_json_results;
pub use pocket_writer::write_pdb_with_pockets;
pub use provenance::write_provenance_metadata;
pub use publication::{
    write_publication_json, BoundingBox, DetectionMetadata, PocketGeometry, PocketOutput,
    PocketScores, PrismOutput, ProteinMetadata, RuntimeMetadata, PRISM_VERSION, SCHEMA_VERSION,
};
pub use visualization::write_pymol_script;

use crate::pocket::Pocket;
use crate::structure::ProteinStructure;
use std::path::Path;
use std::time::Instant;

/// Convenience wrapper to write pockets to PDB file with provenance
pub fn write_pockets_pdb(
    pockets: &[Pocket],
    structure: &ProteinStructure,
    path: &Path,
) -> std::io::Result<()> {
    let start = Instant::now();
    write_pdb_with_pockets(path, structure, pockets)?;
    let _ = write_provenance_metadata(path, start, 1);
    Ok(())
}

/// Convenience wrapper to write pockets to JSON file with provenance
pub fn write_pockets_json(
    pockets: &[Pocket],
    structure: &ProteinStructure,
    path: &Path,
) -> std::io::Result<()> {
    let start = Instant::now();
    write_json_results(path, structure, pockets)?;
    let _ = write_provenance_metadata(path, start, 1);
    Ok(())
}
