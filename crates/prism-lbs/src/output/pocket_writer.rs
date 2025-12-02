//! PDB writer with pocket annotations (placeholder)

use crate::pocket::Pocket;
use crate::structure::ProteinStructure;
use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;

pub fn write_pdb_with_pockets(
    path: &Path,
    structure: &ProteinStructure,
    pockets: &[Pocket],
) -> Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "REMARK  PRISM-LBS pockets={}", pockets.len())?;
    for atom in &structure.atoms {
        writeln!(
            f,
            "{:<6}{:>5} {:<4}{:1}{:<3} {:1}{:>4}{:1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
            if atom.is_hetatm { "HETATM" } else { "ATOM" },
            atom.serial,
            atom.name,
            atom.alt_loc.unwrap_or(' '),
            atom.residue_name,
            atom.chain_id,
            atom.residue_seq,
            atom.insertion_code.unwrap_or(' '),
            atom.coord[0],
            atom.coord[1],
            atom.coord[2],
            atom.occupancy,
            atom.b_factor,
            atom.element
        )?;
    }
    writeln!(f, "END")?;
    Ok(())
}
