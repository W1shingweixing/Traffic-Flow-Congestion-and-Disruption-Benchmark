# Global Causal Analysis (Task 3)

System-wide causal discovery over the NSW TfNSW 2025 benchmark, comparing a
NOTEARS-style joint estimator (**MM-DAG-lite**) against **per-node Lasso** on
the same standardised variable matrix.

Owner: Xinyue Zhang (z5546556) — DATA5925 Group 14
Corresponds to thesis Sections 3.3, 3.6 and 4.5.

---

## Contents

| File | Description |
|---|---|
| `03_prepare_global_causal.ipynb` | Builds the daily station-level variable matrix from the hourly benchmark data |
| `04_global_causal_analysis_MM_DAG_vs_Lasso.ipynb` | Fits both methods, runs the robustness checks and the post-estimation audits |
| `global_X.npy` | Variable matrix, 24,455 × 26, float32 |
| `global_variable_matrix.csv` | Same rows as `global_X.npy`, with `station_id` and `date` attached |
| `variable_names.csv` | 26 variable names with their category (`meta` / `incident` / `traffic`) |
| `global_case_summary.json` | Station, day, sample and variable counts |
| `dag_side_by_side.png` | Thesis Fig. 4.3 — the two discovered graphs |
| `regularisation_stability.png` | Thesis Fig. 4.4 — edge selection frequency across the penalty sweep |
| `factual_explanation_panels.png` | Thesis Fig. 4.5 — mean outcome by hour of day under different source levels |
| `Global_Causal_Analysis_Info.docx` | Working notes |

---

## How to run

Both notebooks were written for Google Colab. Run `03` first, then `04`.

### Notebook 03 — data preparation

Upload before running:

```
adj_matrix_67.npy
adj_matrix_67_binary.npy
node_order_67.npy
centrality_rankings.csv
final_incidents_with_weather.csv
final_hourly_flow_allfeature_with_timefeat.csv
```

Writes to `data_global_causal/`: `global_X.npy`, `global_variable_matrix.csv`,
`variable_names.csv`, `global_case_summary.json`.

### Notebook 04 — analysis

Upload the four outputs of `03`, plus
`final_hourly_flow_allfeature_with_timefeat.csv` and
`final_incidents_with_weather.csv` again (Section 13 rebuilds the hourly panel
for the factual explanation plots and the timestamp audit).

Writes to `results/global_causal_compare/`: `scorecard.csv`, `edges_mmdag.csv`,
`edges_lasso.csv`, `edge_robustness_summary.csv`, the three PNGs,
`factual_explanation_profiles.csv`, `factual_effect_sizes.csv`.

Runtime is dominated by Section 9 (leave-station-out), which refits MM-DAG-lite
12 times.

---

## Pipeline

**Aggregation.** Hourly flow is aggregated to one observation per station-day
(41,975 rows), giving `mean_flow_hourly`, `is_major_incident`, `impact_hours`,
`precipitation_daily` and `is_weekend`. Incident records are counted per
station-day per hazard type and pivoted to 18 columns (1,683 non-empty
station-days), then left-joined and zero-filled. Centrality scores are joined on
`station_id`.

**Row drop.** Centrality exists only for the 67 stations in the giant component
of the road network graph, so the 48 stations outside it carry NaN and are
dropped: **17,520 rows (41.7%) removed, leaving 24,455 = 67 × 365**. This is the
only source of NaN in the feature columns.

**Alignment guarantee.** `global_variable_matrix.csv` is saved *after* the
`dropna`, with `station_id` and `date` attached, so it is row-aligned with
`global_X.npy`. This matters: the leave-station-out check in notebook 04 maps
every row of `X` back to its station via this file. An earlier version saved the
pre-`dropna` frame, which had more rows and silently broke the mapping.

**Variables (p = 26).** 5 meta-features (`is_weekend`, `precipitation_daily`,
`degree_cent`, `between_cent`, `close_cent`), 18 incident-type counts, 3 traffic
statistics (`mean_flow_hourly`, `is_major_incident`, `impact_hours`).

**Structural constraints.** Meta-features cannot be targets; traffic statistics
cannot be sources. Both methods use the same forbidden mask and the same final
edge threshold (`|W| > 0.01`), so the comparison is like-for-like. MM-DAG-lite
enforces the constraints inside the objective; Lasso applies them post-hoc.

---

## Results

### Method comparison

| Criterion | MM-DAG-lite | Lasso (per-node) |
|---|---|---|
| Produces a valid DAG (acyclic) | No | **Yes** |
| Constraints enforced in-optimisation | **Yes** | No (post-hoc) |
| Edges (full data) | 175 | 29 |
| Edges agreed by both methods | 19 | 19 |
| Mean R² (non-meta nodes) | −0.141 | **0.076** |
| Reg.-path stability ratio | 0.09 (29/320) | **0.63** (19/30) |
| Leave-station-out mean Jaccard | 0.172 | **0.660** |
| Edges stable in ≥70% of resamples | 16 | **27** |

MM-DAG-lite drives the continuous acyclicity penalty to ĥ ≈ 1.5 × 10⁻⁷, but the
thresholded graph still contains cycles. Jaccard overlap between the two edge
sets is 0.103. **Lasso is adopted as the primary estimator**; MM-DAG-lite is
retained as an independent robustness check.

The sparsity diagnostic explains why. Only `breakdown` (3.61%) and `crash`
(2.18%) are active on more than 1% of station-days; `grass_fire`,
`heavy_traffic`, `holiday_traffic` and `late_finishing_roadwork` are effectively
constant columns. The joint objective is poorly conditioned under this regime.

### Eight edges surviving all checks under both methods

```
crash                      -> is_major_incident
hazard                     -> is_major_incident
building_fire              -> is_major_incident
special_event              -> impact_hours
hazard                     -> impact_hours
traffic_lights_blacked_out -> impact_hours
changed_traffic_conditions -> impact_hours
between_cent               -> mean_flow_hourly
```

**Sign disagreement.** "Agreed" means both methods place an edge at the same
(i, j) position — not that they agree on its sign. Four of the eight carry
opposite signs, most severely `crash -> is_major_incident` (−0.546 under
MM-DAG-lite, +0.884 under Lasso). MM-DAG-lite's negative mean R² means its
coefficients should not be read as effect estimates at all. Only the Lasso signs
are interpreted.

---

## Post-estimation audits

Both audits found defects that **every stability check passed over**. This is
the main methodological contribution of the task.

### Variable-construction audit

Seven of the eight surviving edges recover constructed relations rather than
mechanisms:

- `is_major_incident` is a deterministic relabelling of `hazard_type`, so any
  hazard-count → `is_major_incident` edge recovers a definitional mapping.
- `impact_hours` is the hourly duration expansion of the same incident records
  that generate the incident-count columns.

Only `between_cent -> mean_flow_hourly` links variables built from independent
sources (road network graph vs. detector table). It is also the weakest edge by
magnitude (ŵ = 0.066) — the strongest edges are strong because they are
constructed.

Effect size behind that edge: hub tertile 427.7 veh/h vs. fringe 248.5 veh/h
(+72.1%, d = 0.311, p < 0.001), but the middle tertile (246.8 veh/h) is
indistinguishable from fringe (p = 0.26). Non-monotonic — it separates
bottlenecks from everything else rather than grading with centrality.

### Timestamp alignment audit

The incident and flow hour-of-day profiles are anti-correlated as merged
(r = −0.782) and only align after a +10 h circular shift (r = +0.965).

Splitting by daylight saving confirms a timezone mismatch rather than a
reporting delay — a constant delay cannot shift by exactly one hour at the DST
boundaries:

| Period | DST | Expected | Best shift | Corr. | n |
|---|---|---|---|---|---|
| Jan – 5 Apr | Yes | +11 h | +11 h | +0.948 | 870 |
| 6 Apr – 4 Oct | No | +10 h | +10 h | +0.969 | 1,345 |
| 5 Oct – Dec | Yes | +11 h | +11 h | +0.872 | 982 |

Consequences on the 3,197 matched incidents:

- **1,173 (36.7%) assigned to the wrong local day**
- Modal incident hour moves from 05:00 (as merged) to 15:00 (local)
- Wrong-day rate varies by hazard type: 27.9% (changed traffic conditions) to
  47.9% (adverse weather) among the well-populated categories
- Net distortion of the weekend count is only −10 station-days, which is why
  daily aggregation absorbs most of it and the graph stays stable

**Fix:** `tz_convert` upstream of incident-to-station-hour matching — *not* a
fixed `Timedelta`. Not applied in the current results; the reported edges are
attenuated accordingly.

---

## Known limitations

- The identification assumptions (causal sufficiency, correct structural form,
  acyclicity, faithfulness, correct orientation) cannot be verified from this
  data. Stability under resampling is necessary but not sufficient.
- A 36.7% misdating rate left every stability diagnostic unchanged. Stability is
  not validity.
- MM-DAG-lite does not reach a strictly acyclic solution under this sparsity.
- The timezone fix is diagnosed but not yet applied to the pipeline.

---

## Attribution

`MM-DAG-lite` is adapted from `MMDAG.py` released with the TraffiDent paper
(Gou et al., NeurIPS 2025 Datasets and Benchmarks Track), reduced to a
single-task NOTEARS-style estimator. The Lasso pipeline, both robustness checks
and both audits are original to this project.
