# Revised Core Dataset for Six Baseline Models

This folder contains the final shared inputs for:

- Decision Tree
- Random Forest
- XGBoost
- LSTM
- 1D-CNN
- Lightweight Transformer

## Files

1. `canonical_dataset_revised.csv`  
   The cleaned modelling samples and fixed seven-hour traffic windows.

2. `grouped_random_split_manifest_revised.csv`  
   Frozen train/validation/test assignments based on `split_group_id`.

3. `spatial_split_manifest_revised.csv`  
   Frozen train/validation/test assignments based on station-level separation.

## Final sample counts

| Target class | Samples |
|---|---:|
| NORMAL | 3,581 |
| ACCIDENT | 505 |
| OTHER_DISRUPTION | 1,179 |
| **Total** | **5,265** |

Incident samples: 1,684  
Normal controls: 3,581

## Revisions applied

### 1. Duplicate removal

- Full duplicated rows: **0**
- Duplicated `event_sample_id`: **0**
- Semantic duplicates based on station, anchor time, window, label, and seven flow values: **0**
- Incident identifiers were normalised before deduplication so equivalent forms such as `249280` and `249280.0` are not treated as separate incidents.

### 2. Overlapping incident windows

Each sample uses the seven-hour window:

`t-3, t-2, t-1, t0, t+1, t+2, t+3`

- Overlapping windows with **different target classes** were removed because their labels were ambiguous.
- Final cross-class overlapping pairs: **0**.
- Overlapping windows with the **same target class** were retained as valid distinct incidents.
- Final same-class overlapping pairs: **51**.
- Same-class overlapping incidents remain in the same `split_group_id`, preventing train/validation/test leakage.

### 3. All-zero windows

All-zero traffic windows are retained in the primary dataset because zero values cannot be confirmed as missing observations. The column `all_zero_flow_window` is included for auditing and sensitivity analysis only. It should not be used as a main predictor.

### 4. Frozen splits

#### Grouped-random split

| Partition | Samples |
|---|---:|
| Train | 3,685 |
| Validation | 784 |
| Test | 796 |

No `split_group_id` occurs in more than one partition.

#### Spatial split

| Partition | Samples |
|---|---:|
| Train | 3,703 |
| Validation | 806 |
| Test | 756 |

No station occurs in more than one partition.

## How to use

For each model:

1. Load `canonical_dataset_revised.csv`.
2. Merge it with one manifest using `event_sample_id`.
3. Train on `train`, tune or apply early stopping on `validation`, and evaluate once on `test`.
4. Repeat separately for the grouped-random and spatial manifests.
5. Do not regenerate controls or splits for individual models.

## Model inputs

For tree-based models, derive tabular features from the seven flow columns, such as log flow, relative changes, mean shifts, slopes, standard deviation, range, maximum drop, and maximum rise.

For LSTM, 1D-CNN, and Transformer, use the seven flow values as a sequence. Fit all scalers using the training partition only.

Do not use identifiers, labels, `split_group_id`, `sample_type`, or `all_zero_flow_window` as primary predictors.
