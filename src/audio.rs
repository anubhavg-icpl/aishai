use hound::WavReader;
use std::path::Path;

pub struct AudioFile {
    pub name: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,    // original channel count before mixdown
    pub bit_depth: u16,   // bits per sample from spec
}

pub fn load_wav(path: &Path) -> anyhow::Result<AudioFile> {
    let mut reader = WavReader::open(path).map_err(|e| {
        // hound rejects 64-bit float WAV with "bits per sample is not 32" before
        // we can inspect the spec, so enrich that error here.
        if e.to_string().contains("bits per sample") {
            anyhow::anyhow!(
                "64-bit float WAV not supported — convert with ffmpeg:\n  \
                 ffmpeg -i \"{}\" -acodec pcm_f32le output.wav",
                path.display()
            )
        } else {
            anyhow::Error::from(e)
        }
    })?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            if !matches!(spec.bits_per_sample, 8 | 16 | 24 | 32) {
                anyhow::bail!(
                    "unsupported bit depth {}. Supported: 8, 16, 24, 32-bit PCM; 32-bit float",
                    spec.bits_per_sample
                );
            }
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    // Mix down to mono by averaging channels
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok(AudioFile {
        name: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        samples: mono,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        bit_depth: spec.bits_per_sample,
    })
}

pub fn frames(samples: &[f32], size: usize, hop: usize) -> Vec<Vec<f32>> {
    if samples.len() < size {
        return vec![];
    }
    samples
        .windows(size)
        .step_by(hop)
        .map(|w| w.to_vec())
        .collect()
}
