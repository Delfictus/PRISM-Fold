use clap::{ArgAction, Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};

use prism_lbs::{LbsConfig, OutputConfig, OutputFormat, PrismLbs, ProteinStructure};

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
    let structure = ProteinStructure::from_pdb_file(&cli.input)?;
    let predictor = PrismLbs::new(config.clone())?;
    let pockets = predictor.predict(&structure)?;

    let base = resolve_output_base(&cli.output, &cli.input);
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
                prism_lbs::output::write_pdb_with_pockets(&out_path, &structure, &pockets)?
            }
            OutputFormat::Json => {
                prism_lbs::output::write_json_results(&out_path, &structure, &pockets)?
            }
            OutputFormat::Csv => {}
        }
    }
    if config.output.include_pymol_script {
        let mut pymol_path = base.clone();
        pymol_path.set_extension("pml");
        ensure_parent_dir(&pymol_path)?;
        prism_lbs::output::write_pymol_script(&pymol_path, pockets.len())?;
    }
    Ok(())
}

fn run_batch(cli: &Cli, config: LbsConfig, parallel: usize) -> anyhow::Result<()> {
    if cli.output.is_dir() || cli.output.extension().is_none() {
        fs::create_dir_all(&cli.output)?;
    }

    let mut results = Vec::new();
    for entry in std::fs::read_dir(&cli.input)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("pdb") {
                    results.push(path);
                }
            }
        }
    }

    let predictor = PrismLbs::new(config.clone())?;
    results.chunks(parallel.max(1)).for_each(|batch| {
        for path in batch {
            if let Ok(structure) = ProteinStructure::from_pdb_file(path) {
                if let Ok(pockets) = predictor.predict(&structure) {
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
                                }
                                OutputFormat::Json => {
                                    let _ = prism_lbs::output::write_json_results(
                                        &out_path, &structure, &pockets,
                                    );
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
                    }
                }
            }
        }
    });
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
