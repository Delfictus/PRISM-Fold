//! JSON exporter for pocket predictions

use crate::pocket::Pocket;
use crate::structure::ProteinStructure;
use serde::Serialize;
use std::fs::File;
use std::io::Result;
use std::path::Path;

#[derive(Serialize)]
struct PocketResult<'a> {
    structure: &'a str,
    pockets: &'a [Pocket],
}

pub fn write_json_results(
    path: &Path,
    structure: &ProteinStructure,
    pockets: &[Pocket],
) -> Result<()> {
    let f = File::create(path)?;
    let res = PocketResult {
        structure: structure.title.as_str(),
        pockets,
    };
    serde_json::to_writer_pretty(f, &res)?;
    Ok(())
}
