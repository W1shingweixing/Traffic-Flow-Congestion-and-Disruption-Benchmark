"""
Post-Incident Traffic Forecasting — LSTM Baseline Model
=======================================================
Task 1: Predict traffic flow 1h–6h after an incident, given 24h of history.
Supports three strategies in a single script:
  Strategy 1: General        — no incident information
  Strategy 2: +Binary        — incident existence flag (is_incident ∈ {0, 1})
  Strategy 3: +TypeEmb       — incident type learnable embedding (9 categories)

Architecture:
  2-layer LSTM encoder → last hidden state → Linear decoder → 6 outputs.
  Incident features (binary flag or type embedding) are concatenated
  to each timestep's input vector.

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

DATA_DIR = Path("C:/Users/huang/Desktop/traffic_data_24_6")
TRAIN_PATH = DATA_DIR / "train_24_6.csv"
VAL_PATH   = DATA_DIR / "val_24_6.csv"
TEST_PATH  = DATA_DIR / "test_24_6.csv"

NUM_TIMESTEPS  = 24
FEAT_PER_STEP  = 16
NUM_TARGETS    = 6

# Feature indices (0-indexed) for General strategy
GENERAL_FEAT_IDX = [0, 1, 2, 3, 4,
                     8, 9,
                     10,
                     13, 14]
NUM_GENERAL_FEAT = len(GENERAL_FEAT_IDX)   # 10

NUM_INCIDENT_TYPES = 9
TYPE_EMB_DIM       = 16

# LSTM architecture
HIDDEN_DIM    = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
DECODER_DIM   = 64   # intermediate linear layer after LSTM

# Training
BATCH_SIZE    = 64
MAX_EPOCHS    = 200
PATIENCE      = 15
LEARNING_RATE = 1e-3
SEEDS         = [42, 123, 777]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. Dataset Class
# ============================================================================

class TrafficWindowDataset(Dataset):
    """Loads the 390-column window CSV and returns per-timestep features."""
    def __init__(self, csv_path, strategy="general"):
        self.strategy = strategy.lower()
        raw = pd.read_csv(csv_path, header=None, dtype=np.float32).values
        num_cols = raw.shape[1]

        self.input_flat = raw[:, :NUM_TIMESTEPS * FEAT_PER_STEP]
        self.input_3d   = self.input_flat.reshape(-1, NUM_TIMESTEPS, FEAT_PER_STEP)
        self.targets    = raw[:, num_cols - NUM_TARGETS:]

        self.impact_seq = self.input_3d[:, 0, 5].copy()
        self.inc_type   = self.input_3d[:, 0, 12].astype(np.int64)

    def __len__(self):
        return self.input_flat.shape[0]

    def __getitem__(self, idx):
        # General features per timestep → shape (24, 10)
        # Use two-step indexing to avoid numpy advanced-indexing axis reorder
        x_gen_seq = self.input_3d[idx][:, GENERAL_FEAT_IDX].astype(np.float32)

        is_incident = float(self.impact_seq[idx] >= 0.0)
        type_idx    = self.inc_type[idx]
        y           = self.targets[idx].astype(np.float32)

        return {
            "x_gen_seq": torch.from_numpy(x_gen_seq),       # (24, 10)
            "is_inc":    torch.tensor(is_incident, dtype=torch.float32),
            "x_type_idx":torch.tensor(type_idx, dtype=torch.long),
            "y":         torch.from_numpy(y),               # (6,)
        }


# ============================================================================
# 2. LSTM Model
# ============================================================================

class LSTMForecaster(nn.Module):
    """
    2-layer LSTM.
    At each timestep t, the input vector is:
      "general": [flow, weather, time_feats]                  → 10-d
      "binary":  [flow, weather, time_feats, is_incident]     → 11-d
      "typeemb": [flow, weather, time_feats, type_emb(16)]    → 26-d
    The last hidden state is decoded via Linear(128→64→6).
    """

    def __init__(self, strategy="general"):
        super().__init__()
        self.strategy = strategy.lower()

        # Per-timestep feature dimension
        self.gen_dim    = NUM_GENERAL_FEAT                   # 10
        self.binary_dim = NUM_GENERAL_FEAT + 1               # 11
        self.type_dim   = NUM_GENERAL_FEAT + TYPE_EMB_DIM    # 26

        if self.strategy == "typeemb":
            self.type_emb = nn.Embedding(NUM_INCIDENT_TYPES,
                                         TYPE_EMB_DIM,
                                         padding_idx=0)
            lstm_in = self.type_dim
        elif self.strategy == "binary":
            lstm_in = self.binary_dim
        else:
            lstm_in = self.gen_dim

        self.lstm = nn.LSTM(input_size=lstm_in,
                            hidden_size=HIDDEN_DIM,
                            num_layers=NUM_LAYERS,
                            batch_first=True,
                            dropout=DROPOUT)

        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, DECODER_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DECODER_DIM, NUM_TARGETS),
        )

    def forward(self, batch):
        x_gen_seq = batch["x_gen_seq"].to(DEVICE)          # (B, 24, 10)
        B, T, _ = x_gen_seq.shape

        if self.strategy == "general":
            lstm_input = x_gen_seq

        elif self.strategy == "binary":
            is_inc = batch["is_inc"].to(DEVICE).unsqueeze(-1).unsqueeze(-1)
            # is_inc: (B,) → (B, 1, 1) → broadcast to (B, 24, 1)
            flag = is_inc.expand(B, T, 1)
            lstm_input = torch.cat([x_gen_seq, flag], dim=-1)    # (B, 24, 11)

        elif self.strategy == "typeemb":
            type_idx = batch["x_type_idx"].to(DEVICE)
            is_inc   = batch["is_inc"].to(DEVICE)
            emb = self.type_emb(type_idx)                        # (B, 16)
            emb = emb * is_inc.unsqueeze(-1)                     # mask non-incident
            emb = emb.unsqueeze(1).expand(B, T, TYPE_EMB_DIM)   # (B, 24, 16)
            lstm_input = torch.cat([x_gen_seq, emb], dim=-1)    # (B, 24, 26)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        out, (h_n, c_n) = self.lstm(lstm_input)
        last_hidden = out[:, -1, :]                              # (B, 128)
        return self.decoder(last_hidden)


# ============================================================================
# 3. Training & Evaluation
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
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        pred = model(batch)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch["y"].cpu().numpy())

    preds   = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mae  = np.mean(np.abs(preds - targets), axis=0)
    rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((targets - preds) / (targets + 1e-8))
        ape = np.where(targets == 0.0, 0.0, ape)
        mape = np.mean(ape, axis=0) * 100.0

    return {
        "mae_1h":  float(mae[0]),   "mape_1h":  float(mape[0]),  "rmse_1h":  float(rmse[0]),
        "mae_3h":  float(mae[2]),   "mape_3h":  float(mape[2]),  "rmse_3h":  float(rmse[2]),
        "mae_6h":  float(mae[5]),   "mape_6h":  float(mape[5]),  "rmse_6h":  float(rmse[5]),
        "mae_avg": float(np.mean(mae)),
        "mape_avg":float(np.mean(mape)),
        "rmse_avg":float(np.mean(rmse)),
    }


def run_experiment(strategy, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TrafficWindowDataset(TRAIN_PATH, strategy)
    val_ds   = TrafficWindowDataset(VAL_PATH,   strategy)
    test_ds  = TrafficWindowDataset(TEST_PATH,  strategy)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model     = LSTMForecaster(strategy).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    best_val_mae = float("inf")
    best_state   = None
    no_improve   = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)

        if epoch % 2 == 1 or epoch <= 3:
            print(f"    Epoch {epoch:3d} | Val MAE {val_metrics['mae_avg']:.4f} (best {best_val_mae:.4f})")

        if val_metrics["mae_avg"] < best_val_mae:
            best_val_mae = val_metrics["mae_avg"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val     = val_metrics.copy()
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    train_metrics = evaluate(model, train_loader)

    return {"seed": seed, "train": train_metrics,
            "val": best_val, "test": test_metrics}


# ============================================================================
# 4. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LSTM Model for Post-Incident Traffic Forecasting")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["general", "binary", "typeemb", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--save_dir", type=str,
                        default="C:/Users/huang/Desktop/cloude code/results")
    args = parser.parse_args()

    strategies = (["general", "binary", "typeemb"]
                  if args.strategy == "all" else [args.strategy])

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
            print(f"    Val MAE avg: {result['val']['mae_avg']:.4f}"
                  f"  | Test MAE avg: {result['test']['mae_avg']:.4f}")

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
                "mae_1h": test_mae_1h, "mae_3h": test_mae_3h,
                "mae_6h": test_mae_6h, "mae_avg": test_mae_avg,
                "mape_1h": test_mape_1h, "mape_3h": test_mape_3h,
                "mape_6h": test_mape_6h, "mape_avg": np.mean([test_mape_1h, test_mape_3h, test_mape_6h]),
                "rmse_1h": test_rmse_1h, "rmse_3h": test_rmse_3h,
                "rmse_6h": test_rmse_6h, "rmse_avg": np.mean([test_rmse_1h, test_rmse_3h, test_rmse_6h]),
            }
        }

    print(f"\n{'='*60}")
    print("FINAL RESULTS — LSTM Model")
    print(f"{'='*60}")
    print(f"{'Strategy':<12} {'MAE-1h':>8} {'MAE-3h':>8} {'MAE-6h':>8} {'MAE-avg':>8} {'MAPE-1h':>8} {'MAPE-3h':>8} {'MAPE-6h':>8} {'RMSE-1h':>8} {'RMSE-3h':>8} {'RMSE-6h':>8}")
    print("-" * 104)
    for strat in ["general", "binary", "typeemb"]:
        if strat not in all_results: continue
        s = all_results[strat]["summary"]
        print(f"{strat:<12} {s['mae_1h']:8.4f} {s['mae_3h']:8.4f} "
              f"{s['mae_6h']:8.4f} {s['mae_avg']:8.4f} "
              f"{s['mape_1h']:7.2f}% {s['mape_3h']:7.2f}% {s['mape_6h']:7.2f}% "
              f"{s['rmse_1h']:8.4f} {s['rmse_3h']:8.4f} {s['rmse_6h']:8.4f}")

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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "lstm_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
