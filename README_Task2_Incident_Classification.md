# Task 2 — Incident Classification

## Overview

Task 2 evaluates whether short-term traffic-flow patterns can distinguish three traffic conditions:

- `NORMAL`
- `ACCIDENT`
- `OTHER_DISRUPTION`

Each sample is represented by a seven-hour sequence centred on a reference event time:

\[
t-3,\; t-2,\; t-1,\; t_0,\; t+1,\; t+2,\; t+3
\]

The three observations before \(t_0\) represent pre-event conditions, while the observations after \(t_0\) capture the initial traffic response or recovery period.

## Research Question

> How accurately can traffic-flow patterns classify normal conditions, accidents, and other disruptions, including at previously unseen traffic stations?

## Input Representation

The benchmark-level input is:

\[
X \in \mathbb{R}^{7 \times 2}
\]

with two channels:

1. **Log-transformed traffic flow**
2. **Pre-event-normalized relative change**

The classification target is:

\[
f:\mathbb{R}^{7\times2}
\rightarrow
\{\text{NORMAL},\text{ACCIDENT},\text{OTHER\_DISRUPTION}\}
\]

### Model-specific input processing

- **Decision Tree, Random Forest, and XGBoost** use the flattened two-channel sequence together with summary traffic features.
- **LSTM, 1D-CNN, and Lightweight Transformer** process the temporal sequence directly.
- **Mantis-8M + MLP and MOMENT-1-base + MLP** adapt the seven-timestep sequence to length 512, pass it through a frozen pretrained encoder, and classify the resulting embedding using a supervised MLP head.
  - Mantis embedding: 512 dimensions
  - MOMENT embedding: 768 dimensions

The interpolation to length 512 is an input adaptation only and does not introduce new temporal information.

## Models

Eight models are evaluated:

1. Decision Tree
2. Random Forest
3. XGBoost
4. LSTM
5. 1D-CNN
6. Lightweight Transformer
7. Mantis-8M + MLP
8. MOMENT-1-base + MLP

### Main configurations

| Model | Key configuration |
|---|---|
| Decision Tree | Maximum depth, minimum samples per leaf, minimum samples split, pruning |
| Random Forest | 300 trees, maximum depth 10–20/None, balanced class weight |
| XGBoost | Maximum depth 3–5, learning rate 0.03–0.07, early stopping |
| LSTM | 32–64 hidden units, dropout 0.20–0.40, AdamW |
| 1D-CNN | Conv1D 32→64, kernel size 3, dropout 0.20, AdamW |
| Lightweight Transformer | 1 encoder block, \(d_{\text{model}}=32\)–48, 2–4 heads, FFN 64–96 |
| Mantis-8M + MLP | Frozen Mantis-8M encoder, 512-d embedding, MLP 64/128, dropout 0.20–0.30 |
| MOMENT-1-base + MLP | Frozen MOMENT-1-base encoder, 768-d embedding, MLP 64/128, dropout 0.20–0.30 |

## Data Files

The final workflow uses three separate CSV files:

```text
canonical_dataset_revised.csv
grouped_random_split_manifest_revised.csv
spatial_split_manifest_revised.csv
```

### Canonical dataset

The canonical dataset contains the samples, labels, station identifiers, and seven-hour traffic-flow observations.

Expected traffic-flow columns:

```text
flow_t_minus_3
flow_t_minus_2
flow_t_minus_1
flow_t0
flow_t_plus_1
flow_t_plus_2
flow_t_plus_3
```

Expected target column:

```text
target_class
```

Expected labels:

```text
NORMAL
ACCIDENT
OTHER_DISRUPTION
```

### Split manifests

Two frozen split manifests are used:

- **Grouped-random manifest** keeps related samples or samples from the same window group in the same partition to reduce data leakage.
- **Spatial-holdout manifest** places selected traffic stations exclusively in validation or test partitions to assess generalisation to unseen stations.

The dataset rows, labels, split membership, and test indices remain fixed across models and seeds.

## Evaluation Protocol

### Primary metric

**Macro-F1** is the main metric because the class distribution is imbalanced. Each class contributes equally to the final score.

### Supporting metrics

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score per class
- Confusion matrix

### Seeds

Each model is evaluated using:

```text
42, 52, 62
```

The seeds only affect stochastic model training. They do not change the data partitions.

### Model selection

- Hyperparameters are selected using validation Macro-F1.
- The test set is used only for final evaluation.
- Results are reported as mean ± standard deviation across the three seeds.
- Early stopping is applied to XGBoost and neural models.
- For Mantis and MOMENT, only the supervised MLP head is trained; the pretrained encoder remains frozen.

## Final Results

### Overall Macro-F1

| Model | Grouped-random Macro-F1 | Spatial-holdout Macro-F1 |
|---|---:|---:|
| Decision Tree | 0.310 ± 0.017 | 0.267 ± 0.003 |
| Random Forest | 0.344 ± 0.012 | 0.316 ± 0.010 |
| XGBoost | **0.349 ± 0.016** | 0.283 ± 0.005 |
| LSTM | 0.325 ± 0.013 | 0.312 ± 0.026 |
| 1D-CNN | 0.343 ± 0.003 | 0.325 ± 0.014 |
| Lightweight Transformer | 0.319 ± 0.011 | **0.334 ± 0.006** |
| Mantis-8M + MLP | 0.318 ± 0.024 | 0.308 ± 0.006 |
| MOMENT-1-base + MLP | **0.349 ± 0.012** | 0.300 ± 0.027 |

### Spatial generalisation

\[
\Delta_{\text{spatial}}
=
F_{1,\text{grouped}}
-
F_{1,\text{spatial}}
\]

| Model | Grouped F1 | Spatial F1 | \(\Delta_{\text{spatial}}\) |
|---|---:|---:|---:|
| Decision Tree | 0.310 | 0.267 | +0.043 |
| Random Forest | 0.344 | 0.316 | +0.028 |
| XGBoost | 0.349 | 0.283 | +0.066 |
| LSTM | 0.325 | 0.312 | +0.013 |
| 1D-CNN | 0.343 | 0.325 | +0.018 |
| Lightweight Transformer | 0.319 | 0.334 | −0.015 |
| Mantis-8M + MLP | 0.318 | 0.308 | +0.010 |
| MOMENT-1-base + MLP | 0.349 | 0.300 | +0.049 |

## Main Findings

- **XGBoost and MOMENT achieved the best grouped-random Macro-F1**, both approximately 0.349.
- **Lightweight Transformer achieved the best spatial-holdout Macro-F1**, at 0.334 ± 0.006.
- **1D-CNN was competitive under both evaluation settings**, with relatively stable performance.
- **Mantis showed a small spatial generalisation gap**, but its absolute Macro-F1 remained moderate.
- **MOMENT was strong under grouped-random evaluation but degraded under spatial holdout.**
- **ACCIDENT remained the most difficult class** across classical, deep-learning, and foundation-model approaches.
- No model achieved a Macro-F1 substantially above 0.35.
- Increasing model complexity or using pretrained foundation models did not consistently improve performance.
- The primary limitation appears to be the weak class separability of the seven-hour hourly traffic-flow representation.

## Recommended Interpretation

The seven-hour traffic-flow sequence contains useful information for incident classification, but the signal is not sufficiently discriminative to reliably separate `NORMAL`, `ACCIDENT`, and `OTHER_DISRUPTION`.

Strong performance on grouped-random data does not necessarily indicate strong generalisation to unseen traffic stations. The spatial-holdout evaluation is therefore essential for assessing real-world transfer.

Future improvements should prioritise:

- higher temporal resolution
- longer temporal context
- traffic speed and occupancy
- travel-time variables
- weather information
- road characteristics
- incident-related contextual features
- stronger class-imbalance handling

## Suggested Repository Structure

```text
task2-incident-classification/
├── README.md
├── data/
│   ├── canonical_dataset_revised.csv
│   ├── grouped_random_split_manifest_revised.csv
│   └── spatial_split_manifest_revised.csv
├── notebooks/
│   ├── task2_decision_tree.ipynb
│   ├── task2_random_forest.ipynb
│   ├── task2_xgboost.ipynb
│   ├── task2_lstm.ipynb
│   ├── task2_1d_cnn.ipynb
│   ├── task2_lightweight_transformer.ipynb
│   ├── task2_mantis.ipynb
│   └── task2_moment.ipynb
├── results/
│   ├── overall_results.csv
│   ├── per_class_results.csv
│   ├── spatial_generalisation.csv
│   └── confusion_matrices/
└── figures/
    ├── macro_f1_comparison.png
    └── spatial_holdout_confusion_matrix.png
```

## Environment Notes

Most models can be run in the main Python environment. MOMENT may require a separate Python 3.11 environment because some dependencies may not install correctly under Python 3.12.

Example:

```bash
conda create -n moment311 python=3.11 -y
conda activate moment311

python -m pip install --upgrade pip setuptools wheel
pip install momentfm numpy pandas scikit-learn matplotlib torch transformers huggingface_hub
pip install jupyter ipykernel

python -m ipykernel install --user \
  --name moment311 \
  --display-name "Python 3.11 (MOMENT)"
```

## References

1. Gou, X., Li, Z., Lan, T., Lin, J., Li, Z., Zhao, B., Zhang, C., Wang, D., and Zhang, X. *TraffiDent: A Dataset for Understanding the Interplay Between Traffic Dynamics and Incidents*. NeurIPS, 2025.
2. Feofanov, V., Alonso, M., Wen, S., Ilbert, R., Guo, H., Tiomoko, M., Pan, L., Zhang, J., and Redko, I. *Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification*. ICLR, 2025.
3. Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A. *MOMENT: A Family of Open Time-series Foundation Models*. ICML, 2024.
4. Sokolova, M. and Lapalme, G. *A Systematic Analysis of Performance Measures for Classification Tasks*. Information Processing & Management, 2009.
5. Roberts, D. R. et al. *Cross-validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure*. Ecography, 2017.
