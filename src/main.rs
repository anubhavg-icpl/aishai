mod audio;
mod features;
mod pipeline;

use std::path::PathBuf;

fn dominant_band(bands: &[f32; 8]) -> &'static str {
    let (max_i, _) = bands
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    features::BAND_NAMES[max_i]
}

fn band_bar(bands: &[f32; 8]) -> String {
    let max = bands.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
    bands
        .iter()
        .map(|&v| {
            let level = (v / max * 8.0) as usize;
            match level {
                0 => '▁',
                1 => '▂',
                2 => '▃',
                3 => '▄',
                4 => '▅',
                5 => '▆',
                6 => '▇',
                _ => '█',
            }
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);

    let data_dir = PathBuf::from("data");
    if !data_dir.exists() {
        anyhow::bail!("'data/' directory not found — create it and drop .wav files inside");
    }

    let mut wav_paths: Vec<PathBuf> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
        })
        .collect();
    wav_paths.sort();

    if wav_paths.is_empty() {
        anyhow::bail!("No .wav files in data/ — add some audio files and retry");
    }

    println!("╔══════════════════════════════════════════════════╗");
    println!("║         Music Genre Fingerprinter                ║");
    println!("║         WAV → FFT → ICA → KMeans                 ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("Found {} WAV file(s) in data/", wav_paths.len());
    println!("Bands: sub-bass bass low-mid mid upper-mid presence brilliance air");
    println!();

    let mut names: Vec<String> = Vec::new();
    let mut feature_rows: Vec<[f32; 8]> = Vec::new();

    for path in &wav_paths {
        match audio::load_wav(path) {
            Ok(af) => {
                let frames = audio::frames(&af.samples, features::FRAME_SIZE, features::HOP_SIZE);
                if frames.is_empty() {
                    println!(
                        "  [skip] {} — too short (< {} samples)",
                        af.name,
                        features::FRAME_SIZE
                    );
                    continue;
                }
                let feats = features::file_features(&frames, af.sample_rate);
                let bar = band_bar(&feats);
                println!(
                    "  [ok]  {:<28} {}Hz  {}  [{}]",
                    af.name,
                    af.sample_rate,
                    bar,
                    dominant_band(&feats)
                );
                names.push(af.name);
                feature_rows.push(feats);
            }
            Err(e) => {
                let fname = path.file_name().unwrap_or_default().to_string_lossy();
                println!("  [err] {fname}: {e}");
            }
        }
    }

    let n = feature_rows.len();
    if n < 2 {
        anyhow::bail!("Need at least 2 valid WAV files to cluster");
    }

    // Build [n × 8] feature matrix in f64
    let flat: Vec<f64> = feature_rows
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f64))
        .collect();
    let matrix = ndarray::Array2::from_shape_vec((n, 8), flat)?;

    let k_actual = k.min(n);
    println!(
        "\nRunning FastICA ({} components) → KMeans (k={})...\n",
        matrix.ncols().min(n),
        k_actual
    );

    let result = pipeline::run(matrix, k_actual)?;

    // Cluster results with per-file dominant band
    println!("╔══════════════════════════════════════════════════╗");
    println!("║                 CLUSTER RESULTS                  ║");
    println!("╠══════════════════════════════════════════════════╣");

    for cluster in 0..k_actual {
        let members: Vec<(&str, &[f32; 8])> = result
            .labels
            .iter()
            .zip(names.iter())
            .zip(feature_rows.iter())
            .filter(|&((&l, _), _)| l == cluster)
            .map(|((_, name), bands)| (name.as_str(), bands))
            .collect();

        if members.is_empty() {
            continue;
        }

        println!(
            "║  Cluster {:>2}  [cluster dominant: {:>10}]          ║",
            cluster, result.dominant_bands[cluster]
        );
        println!("║  {:<48}  ║", "sub▁ bas▁ lmd▁ mid▁ umd▁ pre▁ bri▁ air▁");
        for (name, bands) in &members {
            let bar = band_bar(bands);
            let dom = dominant_band(bands);
            println!("║    • {:<24} {}  {:>10}  ║", name, bar, dom);
        }
        println!("║                                                  ║");
    }

    println!("╚══════════════════════════════════════════════════╝");
    println!("Tip: `cargo run -- 3` to try k=3 clusters");

    Ok(())
}
