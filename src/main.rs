mod audio;
mod features;
mod pipeline;

use std::path::PathBuf;
use unicode_width::UnicodeWidthStr;

// Inner width of every box row (display columns, not bytes)
const W: usize = 56;

// ── box helpers ────────────────────────────────────────────

fn top() -> String { format!("╔{}╗", "═".repeat(W)) }
fn mid() -> String { format!("╠{}╣", "═".repeat(W)) }
fn bot() -> String { format!("╚{}╝", "═".repeat(W)) }

/// Wrap `s` in box borders, padding to exactly W display columns.
fn row(s: &str) -> String {
    let w = UnicodeWidthStr::width(s);
    let pad = W.saturating_sub(w);
    format!("║{}{}║", s, " ".repeat(pad))
}

/// Centre `s` within W display columns.
fn center(s: &str) -> String {
    let w = UnicodeWidthStr::width(s);
    let total = W.saturating_sub(w);
    let l = total / 2;
    let r = total - l;
    format!("{}{}{}", " ".repeat(l), s, " ".repeat(r))
}

/// Pad `s` on the right to `target` display columns.
fn rpad(s: &str, target: usize) -> String {
    let w = UnicodeWidthStr::width(s);
    format!("{}{}", s, " ".repeat(target.saturating_sub(w)))
}

/// Left-pad `s` to `target` display columns.
fn lpad(s: &str, target: usize) -> String {
    let w = UnicodeWidthStr::width(s);
    format!("{}{}", " ".repeat(target.saturating_sub(w)), s)
}

// ── audio helpers ──────────────────────────────────────────

fn dominant_band(bands: &[f32; 8]) -> &'static str {
    let (i, _) = bands
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    features::BAND_NAMES[i]
}

fn band_bar(bands: &[f32; 8]) -> String {
    let max = bands.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
    bands
        .iter()
        .map(|&v| match (v / max * 8.0) as usize {
            0 => '▁',
            1 => '▂',
            2 => '▃',
            3 => '▄',
            4 => '▅',
            5 => '▆',
            6 => '▇',
            _ => '█',
        })
        .collect()
}

// ── main ───────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);

    let data_dir = PathBuf::from("data");
    if !data_dir.exists() {
        anyhow::bail!("'data/' not found — create it and add .wav files");
    }

    let mut wav_paths: Vec<PathBuf> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
        .collect();
    wav_paths.sort();

    if wav_paths.is_empty() {
        anyhow::bail!("No .wav files in data/");
    }

    // ── header ─────────────────────────────────────────────
    println!("{}", top());
    println!("{}", row(&center("Music Genre Fingerprinter")));
    println!("{}", row(&center("WAV  ->  FFT  ->  ICA  ->  KMeans")));
    println!("{}", bot());
    println!();
    println!("  {} WAV file(s) found in data/", wav_paths.len());
    println!("  Bands:  sub-bass  bass  low-mid  mid  upper-mid  presence  brilliance  air");
    println!();

    // ── feature extraction ─────────────────────────────────
    let mut names: Vec<String> = Vec::new();
    let mut feature_rows: Vec<[f32; 8]> = Vec::new();

    for path in &wav_paths {
        match audio::load_wav(path) {
            Ok(af) => {
                let frames =
                    audio::frames(&af.samples, features::FRAME_SIZE, features::HOP_SIZE);
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
                let dom = dominant_band(&feats);
                let ch_label = match af.channels {
                    1 => "mono  ",
                    2 => "stereo",
                    _ => "multi ",
                };
                // unicode-width-aware name column (24 display cols)
                let line = format!(
                    "  [ok]  {}  {}kHz  {}b  {}  {}  [{}]",
                    rpad(&af.name, 24),
                    af.sample_rate / 1000,
                    af.bit_depth,
                    ch_label,
                    bar,
                    dom
                );
                println!("{}", line);
                names.push(af.name);
                feature_rows.push(feats);
            }
            Err(e) => {
                println!(
                    "  [err] {}: {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    e
                );
            }
        }
    }

    let n = feature_rows.len();
    if n < 2 {
        anyhow::bail!("Need at least 2 valid WAV files");
    }

    // ── pipeline ───────────────────────────────────────────
    let flat: Vec<f64> = feature_rows
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as f64))
        .collect();
    let matrix = ndarray::Array2::from_shape_vec((n, 8), flat)?;

    let k_actual = k.min(n);
    println!(
        "\n  FastICA ({} components)  ->  KMeans (k={}) ...\n",
        matrix.ncols().min(n),
        k_actual
    );

    let result = pipeline::run(matrix, k_actual)?;

    // ── results box ────────────────────────────────────────
    println!("{}", top());
    println!("{}", row(&center("CLUSTER  RESULTS")));
    println!("{}", mid());

    // column widths (display cols)
    const NAME_W: usize = 22;
    const BAR_W: usize = 8;
    const DOM_W: usize = 10;

    for cluster in 0..k_actual {
        let members: Vec<(&str, &[f32; 8])> = result
            .labels
            .iter()
            .zip(names.iter())
            .zip(feature_rows.iter())
            .filter(|&((&l, _), _)| l == cluster)
            .map(|((_, nm), b)| (nm.as_str(), b))
            .collect();

        if members.is_empty() {
            continue;
        }

        // cluster header row
        let hdr = format!(
            "  Cluster {}  dominant: {}",
            lpad(&cluster.to_string(), 2),
            lpad(&result.dominant_bands[cluster], DOM_W)
        );
        println!("{}", row(&hdr));

        // band label sub-header
        let sub = "    sub   bas   lmd   mid   umd   pre   bri   air";
        println!("{}", row(sub));

        // one row per file
        for (name, bands) in &members {
            let bar = band_bar(bands);
            let dom = dominant_band(bands);
            let line = format!(
                "  * {}  {}  {}",
                rpad(name, NAME_W),
                rpad(&bar, BAR_W),
                lpad(dom, DOM_W)
            );
            println!("{}", row(&line));
        }

        println!("{}", row(""));
    }

    println!("{}", bot());
    println!("\n  Tip: cargo run -- 3   (change cluster count)");

    Ok(())
}
