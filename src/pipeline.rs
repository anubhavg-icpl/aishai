use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_ica::fast_ica::FastIca;
use ndarray::{Array1, Array2, Axis};

use crate::features::BAND_NAMES;

pub struct PipelineResult {
    pub labels: Vec<usize>,
    pub dominant_bands: Vec<String>,
}

pub fn run(features: Array2<f64>, n_clusters: usize) -> anyhow::Result<PipelineResult> {
    let (n_samples, n_features) = features.dim();

    // Z-score normalize each column
    let mean = features.mean_axis(Axis(0)).unwrap();
    let std = features.std_axis(Axis(0), 1.0);
    let mut normalized = features.clone();
    for mut row in normalized.rows_mut() {
        for j in 0..n_features {
            let s = if std[j] < 1e-10 { 1.0 } else { std[j] };
            row[j] = (row[j] - mean[j]) / s;
        }
    }

    // FastICA: decorrelate frequency band features
    let n_components = n_features.min(n_samples);
    let dataset = DatasetBase::from(normalized.clone());
    let ica = FastIca::params().ncomponents(n_components).fit(&dataset)?;
    // predict(&array_ref) → Array2<F> via PredictInplace blanket impl
    let ica_out: Array2<f64> = ica.predict(&normalized);

    // KMeans on ICA-transformed features
    let k = n_clusters.min(n_samples);
    let ica_dataset = DatasetBase::from(ica_out.clone());
    let model = KMeans::params(k)
        .max_n_iterations(200)
        .tolerance(1e-5)
        .fit(&ica_dataset)?;
    // predict(&array_ref) → Array1<usize>
    let labels_arr: Array1<usize> = model.predict(&ica_out);
    let labels: Vec<usize> = labels_arr.iter().copied().collect();

    // Dominant freq band per cluster: average original (unnormalized) features per cluster
    let mut cluster_sums = vec![[0.0f64; 8]; k];
    let mut cluster_counts = vec![0usize; k];
    for (i, &cl) in labels.iter().enumerate() {
        for j in 0..8 {
            cluster_sums[cl][j] += features[[i, j]];
        }
        cluster_counts[cl] += 1;
    }

    let dominant: Vec<String> = (0..k)
        .map(|cl| {
            let n = cluster_counts[cl].max(1) as f64;
            let (max_i, _) = cluster_sums[cl]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    (*a / n)
                        .partial_cmp(&(*b / n))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or((0, &0.0));
            BAND_NAMES[max_i].to_string()
        })
        .collect();

    Ok(PipelineResult {
        labels,
        dominant_bands: dominant,
    })
}
