use spectrum_analyzer::scaling::divide_by_N_sqrt;
use spectrum_analyzer::windows::hann_window;
use spectrum_analyzer::{FrequencyLimit, samples_fft_to_spectrum};

pub const FRAME_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 1024;

const BANDS: [(f32, f32); 8] = [
    (20.0, 60.0),
    (60.0, 250.0),
    (250.0, 500.0),
    (500.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 6000.0),
    (6000.0, 12000.0),
    (12000.0, 20000.0),
];

pub const BAND_NAMES: [&str; 8] = [
    "sub-bass",
    "bass",
    "low-mid",
    "mid",
    "upper-mid",
    "presence",
    "brilliance",
    "air",
];

fn extract_bands(frame: &[f32], sample_rate: u32) -> [f32; 8] {
    let nyquist = sample_rate as f32 / 2.0;
    if nyquist < 25.0 {
        return [0.0; 8];
    }

    let freq_res = sample_rate as f32 / frame.len() as f32;
    let freq_min = freq_res.max(20.0);
    let freq_max = nyquist.min(20000.0);
    if freq_min >= freq_max {
        return [0.0; 8];
    }

    let windowed = hann_window(frame);
    let spectrum = match samples_fft_to_spectrum(
        &windowed,
        sample_rate,
        FrequencyLimit::Range(freq_min, freq_max),
        Some(&divide_by_N_sqrt),
    ) {
        Ok(s) => s,
        Err(_) => return [0.0; 8],
    };

    let mut bands = [0.0f32; 8];
    let mut counts = [0u32; 8];

    for (freq, mag) in spectrum.data() {
        let f = freq.val();
        for (i, &(lo, hi)) in BANDS.iter().enumerate() {
            if lo > nyquist {
                break;
            }
            if f >= lo && f < hi {
                bands[i] += mag.val();
                counts[i] += 1;
                break;
            }
        }
    }

    for i in 0..8 {
        if counts[i] > 0 {
            bands[i] /= counts[i] as f32;
        }
    }
    bands
}

pub fn file_features(frames: &[Vec<f32>], sample_rate: u32) -> [f32; 8] {
    let mut acc = [0.0f32; 8];
    let mut n = 0usize;

    for frame in frames {
        let b = extract_bands(frame, sample_rate);
        if b.iter().any(|&v| v > 1e-12) {
            for i in 0..8 {
                acc[i] += b[i];
            }
            n += 1;
        }
    }

    if n > 0 {
        for v in &mut acc {
            *v /= n as f32;
        }
    }
    acc
}
