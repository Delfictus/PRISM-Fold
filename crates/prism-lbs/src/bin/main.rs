use clap::{ArgAction, Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use prism_lbs::{
    LbsConfig, OutputConfig, OutputFormat, PrecisionMode, PrismLbs, ProteinStructure,
    UnifiedDetector,
    graph::ProteinGraphBuilder,
    output::{write_publication_json, write_provenance_metadata},
    pocket::filter_by_mode,
};

#[derive(Parser)]
#[command(name = "prism-lbs")]
#[command(about = "PRISM-LBS: Ligand Binding Site Prediction", long_about = None)]
struct Cli {
    /// Input PDB file or directory
    #[arg(short, long)]
    input: PathBuf,

    /// Output path (file or directory)
    #[arg(short, long)]
    output: PathBuf,

    /// Config TOML file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Disable GPU
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Force GPU for surface/geometry stages
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "cpu_geometry")]
    gpu_geometry: bool,

    /// Force CPU for surface/geometry stages
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "gpu_geometry")]
    cpu_geometry: bool,

    /// GPU device id (uses PRISM_GPU_DEVICE if set)
    #[arg(long, default_value_t = 0, env = "PRISM_GPU_DEVICE")]
    gpu_device: usize,

    /// Directory containing PTX modules (defaults to PRISM_PTX_DIR or target/ptx)
    #[arg(long, value_name = "DIR", env = "PRISM_PTX_DIR")]
    ptx_dir: Option<PathBuf>,

    /// Output formats (comma-separated: pdb,json,pymol)
    #[arg(long)]
    format: Option<String>,

    /// Top N pockets to keep
    #[arg(long)]
    top_n: Option<usize>,

    /// Use unified detector (geometric + softspot cryptic site detection)
    #[arg(long, action = ArgAction::SetTrue)]
    unified: bool,

    /// Use softspot-only detection (cryptic sites only, no geometry)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "unified")]
    softspot_only: bool,

    /// Use publication-ready output format (Nature Communications standard)
    #[arg(long, action = ArgAction::SetTrue)]
    publication: bool,

    /// Precision mode for pocket filtering: high_recall, balanced (default), high_precision
    #[arg(long, default_value = "balanced")]
    precision: String,

    /// Ultra-fast screening: use only mega-fused GPU kernel, skip all CPU geometry
    #[arg(long, action = ArgAction::SetTrue)]
    pure_gpu: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Batch process an input directory
    Batch {
        /// Number of parallel tasks
        #[arg(long, default_value_t = 1)]
        parallel: usize,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let mut config = if let Some(cfg) = &cli.config {
        LbsConfig::from_file(cfg)?
    } else {
        LbsConfig::default()
    };

    if cli.cpu {
        config.use_gpu = false;
        config.graph.use_gpu = false;
    }
    if cli.cpu_geometry {
        config.use_gpu = false;
    } else if cli.gpu_geometry {
        config.use_gpu = true;
    }

    if !cli.cpu {
        // Align graph GPU preference with geometry unless explicitly disabled in config
        config.graph.use_gpu = config.graph.use_gpu || config.use_gpu;
    }

    // Make PTX path/device discoverable for library constructors
    if let Some(ref ptx_dir) = cli.ptx_dir {
        std::env::set_var("PRISM_PTX_DIR", ptx_dir);
    }
    std::env::set_var("PRISM_GPU_DEVICE", cli.gpu_device.to_string());

    if let Some(top) = cli.top_n {
        config.top_n = top;
    }

    #[cfg(feature = "cuda")]
    if cli.pure_gpu {
        config.pure_gpu_mode = true;
    }
    if let Some(ref fmt) = cli.format {
        let fmts = fmt
            .split(',')
            .filter_map(|f| match f.trim().to_ascii_lowercase().as_str() {
                "pdb" => Some(OutputFormat::Pdb),
                "json" => Some(OutputFormat::Json),
                "csv" => Some(OutputFormat::Csv),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !fmts.is_empty() {
            config.output = OutputConfig {
                formats: fmts,
                include_pymol_script: true,
                include_json: true,
            };
        }
    }

    match &cli.command {
        Some(Commands::Batch { parallel }) => run_batch(&cli, config, *parallel),
        None => run_single(&cli, config),
    }
}

fn run_single(cli: &Cli, config: LbsConfig) -> anyhow::Result<()> {
    let start_time = Instant::now();
    let mut structure = ProteinStructure::from_file(&cli.input)?;
    let base = resolve_output_base(&cli.output, &cli.input);

    // Choose detection mode
    if cli.unified || cli.softspot_only {
        // Unified or softspot-only mode
        let structure_name = cli.input.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("structure");

        let unified_output = if cli.softspot_only {
            // Softspot-only: run directly on atoms
            let detector = UnifiedDetector::with_config(
                prism_lbs::unified::UnifiedDetectorConfig {
                    enable_geometric: false,
                    enable_softspot: true,
                    max_pockets: config.top_n,
                    ..Default::default()
                }
            );
            detector.detect_from_atoms(&structure.atoms, structure_name)
        } else {
            // Unified: build graph and run both detectors
            structure.compute_surface_accessibility()?;
            let graph_builder = ProteinGraphBuilder::new(config.graph.clone());
            let graph = graph_builder.build(&structure)?;

            let detector = UnifiedDetector::with_config(
                prism_lbs::unified::UnifiedDetectorConfig {
                    enable_geometric: true,
                    enable_softspot: true,
                    max_pockets: config.top_n,
                    ..Default::default()
                }
            );
            detector.detect(&graph, structure_name)?
        };

        // Write unified JSON output
        let mut json_path = base.clone();
        json_path.set_extension("json");
        ensure_parent_dir(&json_path)?;
        let json_content = serde_json::to_string_pretty(&unified_output)?;
        fs::write(&json_path, &json_content)?;
        let _ = write_provenance_metadata(&json_path, start_time, 1);
        log::info!("Wrote unified results to {:?}", json_path);
    } else if cli.pure_gpu {
        // PURE GPU DIRECT MODE: No graph construction, no CPU geometry
        #[cfg(feature = "cuda")]
        {
            log::info!("PURE GPU DIRECT MODE: Bypassing graph construction entirely");
            let predictor = PrismLbs::new(config.clone())?;
            let pockets = predictor.predict_pure_gpu(&structure)?;
            let processing_time_ms = start_time.elapsed().as_millis() as u64;

            for fmt in &config.output.formats {
                let mut out_path = base.clone();
                match fmt {
                    OutputFormat::Pdb => {
                        out_path.set_extension("pdb");
                    }
                    OutputFormat::Json => {
                        out_path.set_extension("json");
                    }
                    OutputFormat::Csv => {
                        out_path.set_extension("csv");
                    }
                }
                ensure_parent_dir(&out_path)?;
                match fmt {
                    OutputFormat::Pdb => {
                        prism_lbs::output::write_pdb_with_pockets(&out_path, &structure, &pockets)?;
                        let _ = write_provenance_metadata(&out_path, start_time, 1);
                    }
                    OutputFormat::Json => {
                        if cli.publication {
                            write_publication_json(
                                &out_path,
                                &pockets,
                                &structure,
                                processing_time_ms,
                                config.use_gpu,
                                None,
                            )?;
                        } else {
                            prism_lbs::output::write_json_results(&out_path, &structure, &pockets)?;
                        }
                        let _ = write_provenance_metadata(&out_path, start_time, 1);
                    }
                    OutputFormat::Csv => {}
                }
            }
            if config.output.include_pymol_script {
                let mut pymol_path = base.clone();
                pymol_path.set_extension("pml");
                ensure_parent_dir(&pymol_path)?;
                prism_lbs::output::write_pymol_script(&pymol_path, pockets.len())?;
                let _ = write_provenance_metadata(&pymol_path, start_time, 1);
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("--pure-gpu requires CUDA feature to be enabled");
        }
    } else {
        // Standard geometric mode
        let predictor = PrismLbs::new(config.clone())?;
        let raw_pockets = predictor.predict(&structure)?;
        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        // Apply precision filtering based on CLI flag
        let precision_mode = PrecisionMode::from_str(&cli.precision)
            .unwrap_or(PrecisionMode::Balanced);
        let (pockets, filter_stats) = filter_by_mode(raw_pockets, precision_mode);
        log::info!("Precision filter ({}): {} -> {} pockets",
            precision_mode, filter_stats.input_count, filter_stats.output_count);

        for fmt in &config.output.formats {
            let mut out_path = base.clone();
            match fmt {
                OutputFormat::Pdb => {
                    out_path.set_extension("pdb");
                }
                OutputFormat::Json => {
                    out_path.set_extension("json");
                }
                OutputFormat::Csv => {
                    out_path.set_extension("csv");
                }
            }
            ensure_parent_dir(&out_path)?;
            match fmt {
                OutputFormat::Pdb => {
                    prism_lbs::output::write_pdb_with_pockets(&out_path, &structure, &pockets)?;
                    let _ = write_provenance_metadata(&out_path, start_time, 1);
                }
                OutputFormat::Json => {
                    if cli.publication {
                        // Publication-ready format with all required fields
                        write_publication_json(
                            &out_path,
                            &pockets,
                            &structure,
                            processing_time_ms,
                            config.use_gpu,
                            None, // GPU name - could be detected
                        )?;
                    } else {
                        // Legacy format for backward compatibility
                        prism_lbs::output::write_json_results(&out_path, &structure, &pockets)?;
                    }
                    let _ = write_provenance_metadata(&out_path, start_time, 1);
                }
                OutputFormat::Csv => {}
            }
        }
        if config.output.include_pymol_script {
            let mut pymol_path = base.clone();
            pymol_path.set_extension("pml");
            ensure_parent_dir(&pymol_path)?;
            prism_lbs::output::write_pymol_script(&pymol_path, pockets.len())?;
            let _ = write_provenance_metadata(&pymol_path, start_time, 1);
        }
    }
    Ok(())
}

fn run_batch(cli: &Cli, config: LbsConfig, parallel: usize) -> anyhow::Result<()> {
    if cli.output.is_dir() || cli.output.extension().is_none() {
        fs::create_dir_all(&cli.output)?;
    }

    let mut pdb_files = Vec::new();
    for entry in std::fs::read_dir(&cli.input)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            let name = path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            // Support all structure formats: PDB, CIF, and gzipped variants
            if name.ends_with(".pdb") || name.ends_with(".ent")
                || name.ends_with(".cif") || name.ends_with(".mmcif")
                || name.ends_with(".pdb.gz") || name.ends_with(".ent.gz")
                || name.ends_with(".cif.gz") || name.ends_with(".mmcif.gz")
            {
                pdb_files.push(path);
            }
        }
    }

    log::info!("Found {} structure files to process (PDB/CIF)", pdb_files.len());
    let use_publication = cli.publication;
    let use_gpu = config.use_gpu;
    let use_pure_gpu = cli.pure_gpu;
    let precision_mode = PrecisionMode::from_str(&cli.precision)
        .unwrap_or(PrecisionMode::Balanced);

    let predictor = PrismLbs::new(config.clone())?;
    let total = pdb_files.len();
    let mut processed = 0usize;

    pdb_files.chunks(parallel.max(1)).for_each(|batch| {
        for path in batch {
            let start_time = Instant::now();
            if let Ok(structure) = ProteinStructure::from_file(path) {
                // Choose prediction path based on pure_gpu flag
                #[cfg(feature = "cuda")]
                let pockets_result: Result<Vec<_>, _> = if use_pure_gpu {
                    // PURE GPU DIRECT MODE: No graph construction
                    predictor.predict_pure_gpu(&structure).map_err(|e| e.into())
                } else {
                    // Standard mode with graph construction
                    predictor.predict(&structure)
                        .map(|raw| filter_by_mode(raw, precision_mode).0)
                };
                #[cfg(not(feature = "cuda"))]
                let pockets_result: Result<Vec<_>, _> = predictor.predict(&structure)
                    .map(|raw| filter_by_mode(raw, precision_mode).0);

                if let Ok(pockets) = pockets_result {
                    let processing_time_ms = start_time.elapsed().as_millis() as u64;
                    let base = resolve_output_base(&cli.output, path);

                    for fmt in &config.output.formats {
                        let mut out_path = base.clone();
                        match fmt {
                            OutputFormat::Pdb => {
                                out_path.set_extension("pdb");
                            }
                            OutputFormat::Json => {
                                out_path.set_extension("json");
                            }
                            OutputFormat::Csv => {
                                out_path.set_extension("csv");
                            }
                        }
                        if ensure_parent_dir(&out_path).is_ok() {
                            match fmt {
                                OutputFormat::Pdb => {
                                    let _ = prism_lbs::output::write_pdb_with_pockets(
                                        &out_path, &structure, &pockets,
                                    );
                                    let _ = write_provenance_metadata(&out_path, start_time, 1);
                                }
                                OutputFormat::Json => {
                                    if use_publication {
                                        // Publication-ready format with all required fields
                                        let _ = write_publication_json(
                                            &out_path,
                                            &pockets,
                                            &structure,
                                            processing_time_ms,
                                            use_gpu,
                                            None, // GPU name
                                        );
                                    } else {
                                        // Legacy format for backward compatibility
                                        let _ = prism_lbs::output::write_json_results(
                                            &out_path, &structure, &pockets,
                                        );
                                    }
                                    let _ = write_provenance_metadata(&out_path, start_time, 1);
                                }
                                OutputFormat::Csv => {}
                            }
                        }
                    }
                    if config.output.include_pymol_script {
                        let mut pymol_path = base.clone();
                        pymol_path.set_extension("pml");
                        let _ = ensure_parent_dir(&pymol_path).and_then(|_| {
                            prism_lbs::output::write_pymol_script(&pymol_path, pockets.len())
                        });
                        let _ = write_provenance_metadata(&pymol_path, start_time, 1);
                    }

                    processed += 1;
                    if processed % 100 == 0 || processed == total {
                        log::info!("Processed {}/{} structures", processed, total);
                    }
                }
            }
        }
    });

    log::info!("Batch processing complete: {} structures processed", processed);
    Ok(())
}

fn resolve_output_base(output: &Path, input: &Path) -> PathBuf {
    if output.is_dir() || output.extension().is_none() {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        output.join(stem)
    } else {
        output.to_path_buf()
    }
}

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}
