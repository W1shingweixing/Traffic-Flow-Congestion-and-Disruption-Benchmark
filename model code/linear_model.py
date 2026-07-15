"""
Post-Incident Traffic Forecasting — Linear Baseline Model
==========================================================
Task 1: Predict traffic flow 1h–6h after an incident, given 24h of history.
Supports three strategies in a single script:
  Strategy 1: General        — no incident information
  Strategy 2: +Binary        — incident existence flag (is_incident ∈ {0, 1})
  Strategy 3: +TypeEmb       — incident type learnable embedding (9 categories)

Author : [Your Name]
Dataset: TfNSW (New South Wales, Australia), full year 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path

# ============================================================================
# 0. Configuration & Constants
# ============================================================================

# Data paths
DATA_DIR = Path("C:/Users/huang/Desktop/traffic_data_24_6")
TRAIN_PATH = DATA_DIR / "train_24_6.csv"
VAL_PATH   = DATA_DIR / "val_24_6.csv"
TEST_PATH  = DATA_DIR / "test_24_6.csv"

# Dataset structure
NUM_TIMESTEPS  = 24    # 24 hours of history
FEAT_PER_STEP  = 16    # features per timestep
NUM_TARGETS    = 6     # t+1 ... t+6 (1h to 6h horizon)

# Feature indices (1-indexed for clarity, converted to 0-indexed in code)
#  1: total_flow               (dynamic,  continuous)
#  2: temperature_2m           (dynamic,  continuous)
#  3: precipitation            (dynamic,  continuous)
#  4: wind_gusts_10m           (dynamic,  continuous)
#  5: relative_humidity        (dynamic,  continuous)
#  6: impact_sequence_hour     (static,   -1.0 or >=0)
#  7: is_major_incident        (static,   0/1)
#  8: incident_count           (static,   1/2)
#  9: hour_sin                 (dynamic,  [-1, 1])
# 10: hour_cos                 (dynamic,  [-1, 1])
# 11: is_weekend               (static*,  0/1)
# 12: is_holiday               (static*,  0/1)
# 13: incident_type_encoded    (static,   0-8)
# 14: lane_count               (static,   1-2)
# 15: road_functional_hierarchy(static,   0-5)
# 16: distance_to_intersection (static,   0.0)

# Indices for General strategy (1-indexed → 0-indexed conversion in code)
# Skip: col 6 (impact_sequence_hour), col 7 (is_major_incident),
#        col 8 (incident_count), col 13 (incident_type), col 16 (distance=0)
GENERAL_FEAT_IDX = [0, 1, 2, 3, 4,        # dynamic weather+flow  (1-5)
                     8, 9,                 # hour_sin, hour_cos     (9-10)
                     10,                   # is_weekend             (11)
                     13, 14]               # lane_count, road_hier  (14-15)
# total: 10 features per timestep → 240 input dims

# Alternatively, include all useful non-incident features:
# We use a cleaner set: the 5 dynamics + hour_sin/cos + 3 static road/time
NUM_GENERAL_FEAT = len(GENERAL_FEAT_IDX)  # 10

# Incident type vocabulary
NUM_INCIDENT_TYPES = 9   # 0=NO_INCIDENT, 1=CRASH, ..., 8=OTHERS
TYPE_EMB_DIM       = 16  # dimension of the learnable type embedding

# Training hyperparameters
BATCH_SIZE    = 64
MAX_EPOCHS    = 200
PATIENCE      = 15
LEARNING_RATE = 1e-3
SEEDS         = [42, 123, 777]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. Dataset Class
# ============================================================================

class TrafficWindowDataset(Dataset):
    """
    Loads the 390-column window CSV and extracts features for the
    three strategies on-the-fly.

    Each row:
        cols   1 – 384: 24 timesteps × 16 features  (inputs)
        cols 385 – 390: 6 target flow values         (targets)

    Parameters
    ----------
    csv_path  : str or Path
    strategy  : str, one of {"general", "binary", "typeemb"}
    """

    def __init__(self, csv_path, strategy="general"):
        self.strategy = strategy.lower()

        # Read CSV as float32 numpy array (fast, no header row)
        raw = pd.read_csv(csv_path, header=None, dtype=np.float32).values
        num_cols = raw.shape[1]

        # --- Inputs: 24 timesteps × 16 features ---
        self.input_flat = raw[:, :NUM_TIMESTEPS * FEAT_PER_STEP]

        # Reshape to (samples, 24 timesteps, 16 features)
        self.input_3d = self.input_flat.reshape(-1, NUM_TIMESTEPS, FEAT_PER_STEP)

        # --- Targets: t+1 ... t+6  ---
        self.targets = raw[:, num_cols - NUM_TARGETS:]

        # --- Pre-extracted columns for strategy-specific use ---
        # col 6: impact_sequence_hour (index 5), col 13: incident_type_encoded (index 12)
        # Use two-step indexing to avoid numpy advanced-indexing axis reorder
        sample0 = self.input_3d[:, :, :]  # (N, 24, 16)
        self.impact_seq = sample0[:, 0, 5]          # same across all 24 t
        self.inc_type   = sample0[:, 0, 12].astype(np.int64)

    def __len__(self):
        return self.input_flat.shape[0]

    def __getitem__(self, idx):
        """
        Returns a dict with:
          x_general : (240,)  — features for the General strategy
          x_binary  : (241,)  — General + is_incident flag
          x_typeemb : (1,)    — incident_type index (for TypeEmb lookup)
          y         : (6,)    — target flow (t+1 ... t+6)
          is_inc    : (1,)    — binary mask (0 or 1)
        """
        # ---- General features ----
        # Extract GENERAL_FEAT_IDX from each timestep, flatten to 1D
        gen_list = []
        for t in range(NUM_TIMESTEPS):
            gen_list.append(self.input_3d[idx, t, GENERAL_FEAT_IDX])
        x_gen = np.concatenate(gen_list)                     # shape: (240,)

        # ---- Binary flag ----
        is_incident = float(self.impact_seq[idx] >= 0.0)     # 0 or 1

        # Concatenate to form +Binary input
        x_bin = np.append(x_gen, is_incident).astype(np.float32)  # (241,)

        # ---- TypeEmb index ----
        type_idx = self.inc_type[idx]                        # int 0-8

        # ---- Target ----
        y = self.targets[idx].astype(np.float32)             # (6,)

        return {
            "x_general": torch.from_numpy(x_gen),
            "x_binary":  torch.from_numpy(x_bin),
            "x_type_idx": torch.tensor(type_idx, dtype=torch.long),
            "is_inc":    torch.tensor(is_incident, dtype=torch.float32),
            "y":         torch.from_numpy(y),
        }


# ============================================================================
# 2. Linear Model
# ============================================================================

class LinearForecaster(nn.Module):
    """
    A single linear layer mapping (input_dim) → (6 targets).

    Three forward modes are available via `strategy` to keep the
    architecture identical while only changing the input composition:
      "general"  →  x_general only
      "binary"   →  x_general + is_incident flag
      "typeemb"  →  x_general + learned incident-type embedding
    """

    def __init__(self, strategy="general"):
        super().__init__()
        self.strategy = strategy.lower()

        # Input dimensions per strategy
        self.gen_dim    = NUM_TIMESTEPS * NUM_GENERAL_FEAT  # 240
        self.binary_dim = self.gen_dim + 1                   # 241
        self.type_dim   = self.gen_dim + TYPE_EMB_DIM        # 256

        # Type embedding table (for TypeEmb strategy only)
        if self.strategy == "typeemb":
            self.type_emb = nn.Embedding(NUM_INCIDENT_TYPES,
                                         TYPE_EMB_DIM,
                                         padding_idx=0)       # idx 0 → zero vector
            in_dim = self.type_dim
        elif self.strategy == "binary":
            in_dim = self.binary_dim
        else:  # general
            in_dim = self.gen_dim

        # Core linear layer
        self.linear = nn.Linear(in_dim, NUM_TARGETS)

    def forward(self, batch):
        """
        batch: dict returned by TrafficWindowDataset.__getitem__
        Returns: (batch_size, 6)  predicted flows at t+1 ... t+6
        """
        if self.strategy == "general":
            x = batch["x_general"].to(DEVICE)

        elif self.strategy == "binary":
            x = batch["x_binary"].to(DEVICE)

        elif self.strategy == "typeemb":
            x_gen = batch["x_general"].to(DEVICE)           # (B, 240)
            type_idx = batch["x_type_idx"].to(DEVICE)        # (B,)  long
            is_inc   = batch["is_inc"].to(DEVICE)            # (B,)  float

            emb = self.type_emb(type_idx)                    # (B, 16)
            emb = emb * is_inc.unsqueeze(-1)                 # mask: non-incident → zero
            x = torch.cat([x_gen, emb], dim=-1)              # (B, 256)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self.linear(x)


# ============================================================================
# 3. Training & Evaluation Utilities
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        pred = model(batch)
        y    = batch["y"].to(DEVICE)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    """
    Returns per-horizon MAE, MAPE, and RMSE, plus overall averages.
    Horizons: 1h=index 0, 3h=index 2, 6h=index 5
    """
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        pred = model(batch)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch["y"].cpu().numpy())

    preds    = np.concatenate(all_preds, axis=0)       # (N, 6)
    targets  = np.concatenate(all_targets, axis=0)     # (N, 6)

    # Avoid division by zero in MAPE (skip target values that are zero after
    # standardization; in practice scaled targets rarely equal exactly 0.0)
    mae  = np.mean(np.abs(preds - targets), axis=0)
    rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((targets - preds) / (targets + 1e-8))  # add tiny epsilon
        ape = np.where(targets == 0.0, 0.0, ape)            # treat div-by-zero as 0
        mape = np.mean(ape, axis=0) * 100.0                  # percentage

    return {
        "mae_1h":  float(mae[0]),   "mape_1h":  float(mape[0]),  "rmse_1h":  float(rmse[0]),
        "mae_3h":  float(mae[2]),   "mape_3h":  float(mape[2]),  "rmse_3h":  float(rmse[2]),
        "mae_6h":  float(mae[5]),   "mape_6h":  float(mape[5]),  "rmse_6h":  float(rmse[5]),
        "mae_avg": float(np.mean(mae)),
        "mape_avg":float(np.mean(mape)),
        "rmse_avg":float(np.mean(rmse)),
    }


def run_experiment(strategy, seed):
    """
    Train a LinearForecaster for one (strategy, seed) combination.
    Returns the best validation metrics and corresponding test metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loaders
    train_ds = TrafficWindowDataset(TRAIN_PATH, strategy)
    val_ds   = TrafficWindowDataset(VAL_PATH,   strategy)
    test_ds  = TrafficWindowDataset(TEST_PATH,  strategy)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model, optimizer, criterion
    model     = LinearForecaster(strategy).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()   # MAE (consistent with TraffiDent & LargeST)

    best_val_mae = float("inf")
    best_state   = None
    no_improve   = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)

        if epoch % 2 == 1 or epoch <= 3:
            print(f"    Epoch {epoch:3d} | Train Loss {train_loss:.4f} | Val MAE {val_metrics['mae_avg']:.4f} (best {best_val_mae:.4f})")

        if val_metrics["mae_avg"] < best_val_mae:
            best_val_mae = val_metrics["mae_avg"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val     = val_metrics.copy()
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Restore best and evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    train_metrics = evaluate(model, train_loader)

    return {
        "seed":         seed,
        "train":        train_metrics,
        "val":          best_val,
        "test":         test_metrics,
    }


# ============================================================================
# 4. Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Linear Model for Post-Incident Traffic Forecasting")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["general", "binary", "typeemb", "all"],
                        help="Which strategy to run (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Random seeds for training")
    parser.add_argument("--save_dir", type=str,
                        default="C:/Users/huang/Desktop/cloude code/results",
                        help="Directory to save results")
    args = parser.parse_args()

    strategies = (["general", "binary", "typeemb"]
                  if args.strategy == "all"
                  else [args.strategy])

    all_results = {}

    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"Running Strategy: {strat.upper()}")
        print(f"{'='*60}")

        strat_results = []
        for seed in args.seeds:
            print(f"  Seed {seed} ...")
            result = run_experiment(strat, seed)
            strat_results.append(result)
            print(f"    Val  MAE avg: {result['val']['mae_avg']:.4f}"
                  f" | Test MAE avg: {result['test']['mae_avg']:.4f}")

        # Aggregate across seeds
        test_mae_1h  = np.mean([r["test"]["mae_1h"]  for r in strat_results])
        test_mae_3h  = np.mean([r["test"]["mae_3h"]  for r in strat_results])
        test_mae_6h  = np.mean([r["test"]["mae_6h"]  for r in strat_results])
        test_mae_avg = np.mean([r["test"]["mae_avg"] for r in strat_results])
        test_mape_1h = np.mean([r["test"]["mape_1h"] for r in strat_results])
        test_mape_3h = np.mean([r["test"]["mape_3h"] for r in strat_results])
        test_mape_6h = np.mean([r["test"]["mape_6h"] for r in strat_results])
        test_rmse_1h = np.mean([r["test"]["rmse_1h"] for r in strat_results])
        test_rmse_3h = np.mean([r["test"]["rmse_3h"] for r in strat_results])
        test_rmse_6h = np.mean([r["test"]["rmse_6h"] for r in strat_results])

        all_results[strat] = {
            "individual": strat_results,
            "summary": {
                "mae_1h":  test_mae_1h,  "rmse_1h":  test_rmse_1h,
                "mae_3h":  test_mae_3h,  "rmse_3h":  test_rmse_3h,
                "mae_6h":  test_mae_6h,  "rmse_6h":  test_rmse_6h,
                "mae_avg": test_mae_avg, "rmse_avg": np.mean([test_rmse_1h,
                                                               test_rmse_3h,
                                                               test_rmse_6h]),
                "mape_1h": test_mape_1h, "mape_3h": test_mape_3h,
                "mape_6h": test_mape_6h, "mape_avg": np.mean([test_mape_1h, test_mape_3h, test_mape_6h]),
            }
        }

    # ---- Summarise ----
    print(f"\n{'='*60}")
    print("FINAL RESULTS — Linear Model")
    print(f"{'='*60}")
    print(f"{'Strategy':<12} {'MAE-1h':>8} {'MAE-3h':>8} {'MAE-6h':>8} "
          f"{'MAE-avg':>8} {'MAPE-1h':>8} {'MAPE-3h':>8} {'MAPE-6h':>8} {'RMSE-1h':>8} {'RMSE-3h':>8} {'RMSE-6h':>8}")
    print("-" * 104)

    for strat in ["general", "binary", "typeemb"]:
        if strat not in all_results:
            continue
        s = all_results[strat]["summary"]
        print(f"{strat:<12} {s['mae_1h']:8.4f} {s['mae_3h']:8.4f} "
              f"{s['mae_6h']:8.4f} {s['mae_avg']:8.4f} "
              f"{s.get('mape_1h', 0):7.2f}% {s.get('mape_3h', 0):7.2f}% {s.get('mape_6h', 0):7.2f}% "
              f"{s['rmse_1h']:8.4f} {s['rmse_3h']:8.4f} {s['rmse_6h']:8.4f}")

    # ---- Ablation deltas ----
    if "general" in all_results and "binary" in all_results:
        print("\nAblation Deltas:")
        for h in ["1h", "3h", "6h"]:
            g = all_results["general"]["summary"][f"mae_{h}"]
            b = all_results["binary"]["summary"][f"mae_{h}"]
            print(f"  D1 (Binary - General) {h}: {b - g:+.4f}")
        if "typeemb" in all_results:
            for h in ["1h", "3h", "6h"]:
                b = all_results["binary"]["summary"][f"mae_{h}"]
                t = all_results["typeemb"]["summary"][f"mae_{h}"]
                print(f"  D2 (TypeEmb - Binary) {h}: {t - b:+.4f}")

    # ---- Save results ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "linear_results.json", "w") as f:
        # Convert numpy floats to Python floats for JSON
        json.dump(all_results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
