//! mmCIF/CIF file parsing using pdbtbx
//!
//! Provides automatic CIF file support for PRISM-LBS, enabling direct use of
//! CryptoBench, LIGYSIS, and other modern benchmark datasets that use mmCIF format.

use crate::LbsError;
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::{Atom, PdbParseOptions, ProteinStructure, Residue};

/// Supported structure file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureFormat {
    /// Legacy PDB format (.pdb, .ent)
    Pdb,
    /// mmCIF format (.cif, .mmcif)
    Cif,
    /// Gzip-compressed PDB (.pdb.gz, .ent.gz)
    PdbGz,
    /// Gzip-compressed mmCIF (.cif.gz, .mmcif.gz)
    CifGz,
}

impl StructureFormat {
    /// Detect format from file extension (case-insensitive)
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?.to_lowercase();

        if name.ends_with(".cif.gz") || name.ends_with(".mmcif.gz") {
            Some(Self::CifGz)
        } else if name.ends_with(".pdb.gz") || name.ends_with(".ent.gz") {
            Some(Self::PdbGz)
        } else if name.ends_with(".cif") || name.ends_with(".mmcif") {
            Some(Self::Cif)
        } else if name.ends_with(".pdb") || name.ends_with(".ent") {
            Some(Self::Pdb)
        } else {
            None
        }
    }

    /// Check if format requires gzip decompression
    pub fn is_gzipped(&self) -> bool {
        matches!(self, Self::PdbGz | Self::CifGz)
    }

    /// Check if format is CIF-based
    pub fn is_cif(&self) -> bool {
        matches!(self, Self::Cif | Self::CifGz)
    }
}

impl ProteinStructure {
    /// Load structure from any supported format (PDB, CIF, gzipped variants)
    ///
    /// Format is auto-detected from file extension. This is the primary entry point
    /// for loading structures and should be used instead of format-specific methods.
    pub fn from_file(path: &Path) -> Result<Self, LbsError> {
        Self::from_file_with_options(path, PdbParseOptions::default())
    }

    /// Load structure with custom parse options
    pub fn from_file_with_options(path: &Path, options: PdbParseOptions) -> Result<Self, LbsError> {
        let format = StructureFormat::from_path(path).ok_or_else(|| {
            LbsError::PdbParse(format!(
                "Unsupported file format: {}. Supported: .pdb, .cif, .mmcif, .ent (and .gz variants)",
                path.display()
            ))
        })?;

        // Read file contents, decompressing if needed
        let contents = read_file_contents(path, format)?;

        // Parse based on format
        let mut structure = if format.is_cif() {
            parse_cif_contents(&contents, &options)?
        } else {
            Self::from_pdb_str_with_options(&contents, options)?
        };

        structure.pdb_path = Some(path.to_path_buf());
        Ok(structure)
    }

    /// Parse mmCIF from file path directly
    pub fn from_cif_file(path: &Path) -> Result<Self, LbsError> {
        Self::from_cif_file_with_options(path, PdbParseOptions::default())
    }

    /// Parse mmCIF with custom options
    pub fn from_cif_file_with_options(
        path: &Path,
        options: PdbParseOptions,
    ) -> Result<Self, LbsError> {
        let format = StructureFormat::from_path(path).unwrap_or(StructureFormat::Cif);
        let contents = read_file_contents(path, format)?;
        let mut structure = parse_cif_contents(&contents, &options)?;
        structure.pdb_path = Some(path.to_path_buf());
        Ok(structure)
    }

    /// Parse mmCIF from string contents
    pub fn from_cif_str(contents: &str) -> Result<Self, LbsError> {
        Self::from_cif_str_with_options(contents, PdbParseOptions::default())
    }

    /// Parse mmCIF from string with custom options
    pub fn from_cif_str_with_options(
        contents: &str,
        options: PdbParseOptions,
    ) -> Result<Self, LbsError> {
        parse_cif_contents(contents, &options)
    }
}

/// Read file contents, handling gzip compression automatically
fn read_file_contents(path: &Path, format: StructureFormat) -> Result<String, LbsError> {
    let file = File::open(path)?;

    if format.is_gzipped() {
        let decoder = GzDecoder::new(BufReader::new(file));
        let mut contents = String::new();
        BufReader::new(decoder)
            .read_to_string(&mut contents)?;
        Ok(contents)
    } else {
        Ok(std::fs::read_to_string(path)?)
    }
}

/// Parse mmCIF contents into ProteinStructure
fn parse_cif_contents(contents: &str, options: &PdbParseOptions) -> Result<ProteinStructure, LbsError> {
    // Use pdbtbx to parse the CIF file - returns Result<(PDB, Vec<PDBError>), Vec<PDBError>>
    let (pdb, errors) = pdbtbx::open_mmcif_raw(contents, pdbtbx::StrictnessLevel::Loose)
        .map_err(|errors| {
            LbsError::PdbParse(format!(
                "Failed to parse mmCIF file: {}",
                errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ")
            ))
        })?;

    // Log warnings for non-fatal errors
    for error in &errors {
        log::debug!("mmCIF parse warning: {}", error);
    }

    convert_pdbtbx_to_structure(&pdb, options)
}

/// Convert pdbtbx PDB structure to our ProteinStructure format
fn convert_pdbtbx_to_structure(
    pdb: &pdbtbx::PDB,
    options: &PdbParseOptions,
) -> Result<ProteinStructure, LbsError> {
    let mut structure = ProteinStructure::default();
    let mut residue_lookup: HashMap<(usize, char, i32, Option<char>), usize> = HashMap::new();

    // Extract metadata
    if let Some(id) = pdb.identifier.as_ref() {
        structure.pdb_id = Some(id.clone());
    }

    // Look for resolution in REMARK 2
    for (remark_type, remark_text) in pdb.remarks() {
        if *remark_type == 2 {
            if let Some(res) = parse_resolution_from_remark(remark_text) {
                structure.resolution = Some(res);
                break;
            }
        }
    }

    // Process models
    let models: Vec<_> = pdb.models().collect();
    structure.models = models.len().max(1);

    for (model_idx, model) in models.iter().enumerate() {
        // Skip additional models if not including all
        if !options.include_all_models && model_idx > 0 {
            break;
        }

        let model_serial = model_idx + 1;

        // Process all chains
        for chain in model.chains() {
            let chain_id = chain.id().chars().next().unwrap_or('A');

            // Process all residues in chain
            for residue in chain.residues() {
                let res_seq = residue.serial_number() as i32;
                let insertion_code = residue.insertion_code().and_then(|s| s.chars().next());
                let res_name = residue.name().map(|s| s.to_uppercase()).unwrap_or_default();

                // Determine if this is a HETATM residue
                let is_hetatm = !is_standard_residue(&res_name);

                if is_hetatm && !options.include_hetatm {
                    continue;
                }

                // Process conformers (which contain atoms and alternate locations)
                for conformer in residue.conformers() {
                    // Get alternate location from conformer
                    let alt_loc = conformer.alternative_location()
                        .and_then(|s| s.chars().next());

                    if alt_loc.is_some() {
                        structure.has_alternate_locations = true;
                    }

                    // Skip non-A alternate locations unless keeping all
                    if let Some(loc) = alt_loc {
                        if !options.keep_alternate_locations && loc != 'A' {
                            continue;
                        }
                    }

                    // Process atoms in this conformer
                    for atom in conformer.atoms() {
                        let coord = atom.pos();
                        let prism_atom = Atom::from_pdb_fields(
                            atom.serial_number() as u32,
                            atom.name().to_string(),
                            res_name.clone(),
                            chain_id,
                            res_seq,
                            insertion_code,
                            [coord.0, coord.1, coord.2],
                            atom.occupancy(),
                            atom.b_factor(),
                            atom.element().map(|e| e.symbol().to_string()).unwrap_or_default(),
                            alt_loc,
                            model_serial,
                            is_hetatm,
                        );

                        // Get or create residue
                        let residue_key = (model_serial, chain_id, res_seq, insertion_code);
                        let residue_index = *residue_lookup.entry(residue_key).or_insert_with(|| {
                            let mut new_residue = Residue::new(res_name.clone(), chain_id, res_seq, insertion_code);
                            new_residue.model = model_serial;
                            new_residue.is_hetatm = is_hetatm;
                            structure.residues.push(new_residue);
                            let idx = structure.residues.len() - 1;
                            structure
                                .chain_residue_indices
                                .entry(chain_id)
                                .or_default()
                                .push(idx);
                            idx
                        });

                        let atom_index = structure.atoms.len();
                        structure.atoms.push(prism_atom);

                        if let Some(res) = structure.residues.get_mut(residue_index) {
                            res.atom_indices.push(atom_index);
                        }

                        if is_hetatm {
                            structure.hetero_atom_indices.push(atom_index);
                        }
                    }
                }
            }
        }
    }

    // Compute derived properties
    structure.recompute_geometry();
    structure.refresh_residue_properties();

    Ok(structure)
}

/// Check if residue is a standard amino acid
fn is_standard_residue(name: &str) -> bool {
    matches!(
        name,
        "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLN" | "GLU" | "GLY" | "HIS" | "ILE"
            | "LEU" | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL"
            // Common variants
            | "MSE" | "SEC" | "PYL"
            // DNA/RNA bases (treat as standard)
            | "A" | "C" | "G" | "T" | "U" | "DA" | "DC" | "DG" | "DT" | "DU"
    )
}

/// Parse resolution from REMARK 2 text
fn parse_resolution_from_remark(text: &str) -> Option<f64> {
    let upper = text.to_uppercase();
    if let Some(pos) = upper.find("RESOLUTION") {
        let tail = &text[pos + "RESOLUTION".len()..];
        for token in tail.split_whitespace() {
            if let Ok(val) = token.parse::<f64>() {
                return Some(val);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            StructureFormat::from_path(Path::new("test.cif")),
            Some(StructureFormat::Cif)
        );
        assert_eq!(
            StructureFormat::from_path(Path::new("test.mmcif")),
            Some(StructureFormat::Cif)
        );
        assert_eq!(
            StructureFormat::from_path(Path::new("test.cif.gz")),
            Some(StructureFormat::CifGz)
        );
        assert_eq!(
            StructureFormat::from_path(Path::new("test.pdb")),
            Some(StructureFormat::Pdb)
        );
        assert_eq!(
            StructureFormat::from_path(Path::new("test.pdb.gz")),
            Some(StructureFormat::PdbGz)
        );
        assert_eq!(
            StructureFormat::from_path(Path::new("test.txt")),
            None
        );
    }

    #[test]
    fn test_is_standard_residue() {
        assert!(is_standard_residue("ALA"));
        assert!(is_standard_residue("GLY"));
        assert!(!is_standard_residue("HOH"));
        assert!(!is_standard_residue("ATP"));
    }

    // Integration test - requires actual CIF file
    #[test]
    #[ignore]
    fn test_parse_cryptobench_cif() {
        let cif_path = Path::new("../../benchmarks/datasets/cryptobench/cif-files/1a27.cif");
        if cif_path.exists() {
            let structure = ProteinStructure::from_file(cif_path)
                .expect("Failed to parse CryptoBench CIF file");
            assert!(structure.atom_count() > 0);
            assert!(structure.residue_count() > 0);
            println!(
                "Parsed {}: {} atoms, {} residues",
                cif_path.display(),
                structure.atom_count(),
                structure.residue_count()
            );
        }
    }
}
