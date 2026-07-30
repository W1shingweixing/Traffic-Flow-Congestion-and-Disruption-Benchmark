# Post-Incident Traffic Forecasting — Task 1

This repository contains the complete implementation of **Task 1: Post-Incident Traffic Forecasting** using the TfNSW (Transport for New South Wales, Australia) 2025 dataset. The task predicts traffic flow 1, 3, and 6 hours after an incident, given 24 hours of historical observations.

## Repository Structure

```
task1/
├── model code/                  # Python model implementations (7 architectures)
│   ├── linear_model.py          # Linear Regression baseline
│   ├── mlp_model.py             # Multi-Layer Perceptron
│   ├── lstm_model.py            # Long Short-Term Memory network
│   ├── transformer_model.py     # Vanilla Transformer encoder
│   ├── itransformer_model.py    # Inverted Transformer (attention over variates)
│   ├── stgcn_model.py           # Spatio-Temporal Graph Convolutional Network
│   └── stsgcn_model.py          # Spatio-Temporal Synchronous GCN
│
├── time series dataset/         # Input data for sequence models (Linear/MLP/LSTM/Transformer/iTransformer)
│   ├── train_24_6.csv           # Training: 2025-01-01 to 2025-08-08 (~60%)
│   ├── val_24_6.csv             # Validation: 2025-08-09 to 2025-10-23 (~20%)
│   └── test_24_6.csv            # Test: 2025-10-24 to 2025-12-31 (~20%)
│
├── graph dataset/               # Input data for graph-based models (STGCN/STSGCN)
│   ├── graph_train_24_6.csv     # Training: 5,256 samples, 28,290 columns per row
│   ├── graph_val_24_6.csv       # Validation: 1,824 samples
│   └── graph_test_24_6.csv      # Test: 1,651 samples
│
├── adjacent matrix/             # Graph topology files
│   ├── adj_matrix_nsw.npy       # 115×115 weighted adjacency (Haversine km)
│   ├── stsgcn_local_stg_T2.npy  # Local ST graph (230×230, T=2)
│   ├── stsgcn_local_stg_T3.npy  # Local ST graph (345×345, T=3)
│   ├── stsgcn_mask_T2.npy       # Mask matrix (230×230, T=2)
│   ├── stsgcn_mask_T3.npy       # Mask matrix (345×345, T=3)
│   ├── stsgcn_st_embedding_T2.npy  # ST embedding (230×64, T=2)
│   ├── stsgcn_st_embedding_T3.npy  # ST embedding (345×64, T=3)
│   └── graph_target_std.npy     # Target normalization parameters
│
└── results/                     # Experiment outputs
    ├── *_log.txt                # Per-model training logs
    └── *_results.json           # Per-model results (MAE/MAPE/RMSE at 1h/3h/6h)
```

## Task Description

| Dimension | Setting |
|-----------|---------|
| Input | 24 hours of historical traffic flow + weather + road metadata |
| Output | Traffic flow at 1h, 3h, 6h after the anchor timestep |
| Region | New South Wales, Australia (TfNSW) |
| Stations | 115 permanent traffic monitoring stations |
| Period | Full year 2025 |
| Granularity | 1 hour |

## Time Series Dataset Format

Each row in `train_24_6.csv` / `val_24_6.csv` / `test_24_6.csv` contains **390 columns**:

- **384 input columns** = 24 timesteps × 16 features per timestep
- **6 target columns** = traffic flow at t+1 through t+6 (1h to 6h horizon)

The 16 features per timestep follow this layout:

| Offset | Feature | Type |
|:---:|---------|:---:|
| 1 | `total_flow` | Dynamic (Z-score normalized) |
| 2 | `temperature_2m` | Dynamic |
| 3 | `precipitation` | Dynamic |
| 4 | `wind_gusts_10m` | Dynamic |
| 5 | `relative_humidity` | Dynamic |
| 6 | `impact_sequence_hour` | Static per window (−1 = no incident) |
| 7 | `is_major_incident` | Static (0/1) |
| 8 | `incident_count` | Static |
| 9 | `hour_sin` | Dynamic [−1,1] |
| 10 | `hour_cos` | Dynamic [−1,1] |
| 11 | `is_weekend` | Static (0/1) |
| 12 | `is_holiday` | Static (0/1) |
| 13 | `incident_type_encoded` | Static (0-8) |
| 14 | `lane_count` | Static |
| 15 | `road_functional_hierarchy` | Static |
| 16 | `is_anomaly` | Static |

Incident type encoding: `0=NO_INCIDENT, 1=CRASH, 2=BREAKDOWN, 3=HAZARD, 4=ROADWORK, 5=TRAFFIC_CONTROL, 6=ADVERSE_WEATHER, 7=EVENT, 8=OTHERS`

## Graph Dataset Format

Each row in `graph_train_24_6.csv` / `graph_val_24_6.csv` / `graph_test_24_6.csv` contains **28,290 columns**:

- **27,600 input columns** = 115 stations × 24 timesteps × 10 features
- **690 target columns** = 115 stations × 6 hours

All values are Z-score standardized using per-feature statistics computed from the training split only.

The 115 stations correspond to the node order in `adjacent matrix/adj_matrix_nsw.npy`.

## Model Architectures

### 1–5: Sequence Models (use `time series dataset/`)

| # | Model | Input | Spatial Modeling | Description |
|---|-------|-------|:---:|-------------|
| 1 | **Linear** | (240,) flattened | ✗ | Single linear layer baseline |
| 2 | **MLP** | (240,) flattened | ✗ | 3-layer fully connected (512→256→128) |
| 3 | **LSTM** | (24, 10) sequence | ✗ | 2-layer LSTM (hidden=128) |
| 4 | **Transformer** | (24, 10) sequence + PE | ✗ | 3-layer Transformer encoder (d_model=128, nhead=8) |
| 5 | **iTransformer** | (10, 24) inverted | Implicit (attention over feature variates) | Inverted Transformer — self-attention over feature channels |

### 6–7: Graph Models (use `graph dataset/`)

| # | Model | Input | Spatial Modeling | Description |
|---|-------|-------|:---:|-------------|
| 6 | **STGCN** | (115, 24, 10) multi-station | Explicit (Chebyshev GCN + TCN) | Separated spatial-temporal modeling |
| 7 | **STSGCN** | (115, 24, 10) multi-station | Explicit (Synchronous GCN) | Local ST graph (T=2), mask matrix, ST embedding |

## Three Experiment Strategies

Each model is trained with three incremental strategies:

| Strategy | Incident Information Used | Input Delta |
|----------|--------------------------|:---:|
| **General** | None — pure traffic + weather + road features | — |
| **+Binary** | Binary incident flag (is_incident ∈ {0,1}) | +1 dimension |
| **+TypeEmb** | Learnable 16-dimensional embedding of incident category (0–8) | +16 dimensions |

## Running the Models

### Prerequisites

```bash
pip install numpy pandas torch
```

### Sequence Models

```bash
python linear_model.py                    # All strategies, 3 seeds
python linear_model.py --strategy general  # Single strategy
python linear_model.py --seeds 42 123     # Custom seeds
```

### Graph Models

```bash
# Requires KMP_DUPLICATE_LIB_OK on Windows
KMP_DUPLICATE_LIB_OK=TRUE python stgcn_model.py
KMP_DUPLICATE_LIB_OK=TRUE python stsgcn_model.py
```

## Results Summary

Results are saved in `results/` as JSON files with the following structure:

```json
{
  "general": {
    "individual": [{"seed": 42, "train": {...}, "val": {...}, "test": {...}}, ...],
    "summary": {
      "mae_1h": 0.xxxx, "mae_3h": 0.xxxx, "mae_6h": 0.xxxx, "mae_avg": 0.xxxx,
      "mape_1h": xx.xx, "mape_3h": xx.xx, "mape_6h": xx.xx, "mape_avg": xx.xx,
      "rmse_1h": 0.xxxx, "rmse_3h": 0.xxxx, "rmse_6h": 0.xxxx, "rmse_avg": 0.xxxx
    }
  },
  "binary": { ... },
  "typeemb": { ... }
}
```

### Quick Results (MAE avg, Testing Set)

| Model | General | +Binary | +TypeEmb | D1 (Bin−Gen) | D2 (Type−Bin) |
|-------|:-------:|:-------:|:--------:|:------------:|:-------------:|
| Linear | 0.3511 | 0.3504 | 0.3502 | −0.0008 | −0.0002 |
| MLP | 0.2253 | 0.2281 | 0.2264 | +0.0028 | −0.0017 |
| LSTM | 0.1721 | 0.1731 | 0.1733 | +0.0011 | +0.0002 |
| Transformer | 0.1860 | 0.1850 | 0.1830 | −0.0010 | −0.0020 |
| iTransformer | 0.2111 | 0.2078 | 0.2040 | −0.0034 | −0.0038 |
| STGCN | 0.6880 | 0.6880 | 0.6877 | +0.0000 | -0.0003 |
| STSGCN | 0.6598 | 0.6613 | 0.6877 | +0.0016 | +0.0263 |

*\*Results in progress.*

## Key Contributions

1. **First post-incident traffic forecasting benchmark on Australian TfNSW data** — all prior work uses California PeMS.
2. **Three-level incident information ablation** — quantifies the marginal value of incident existence (+Binary) and incident type (+TypeEmb) across 7 architectures.
3. **Mid-to-long horizon prediction (1h–6h)** — 75% of incidents last <1.1 hours; 6h forecasting requires modeling the full recovery process.
4. **Sparse multi-station graph modeling** — compares implicit spatial attention (iTransformer) against explicit graph convolution (STGCN/STSGCN) on a 115-station network spanning hundreds of kilometers.

## Data Source

Traffic data sourced from the Transport for NSW (TfNSW) Open Data portal. Incident records from the TfNSW Live Traffic incident feed. Dataset covers all of 2025 at 1-hour granularity across 115 permanent monitoring stations throughout New South Wales, Australia.

## License

This project is for academic research purposes. Please cite appropriately if using this code or dataset in your work.
