"""
Post-Incident Traffic Forecasting — Transformer Baseline Model
==============================================================
Task 1: Predict traffic flow 1h–6h after an incident, given 24h of history.
Supports three strategies in a single script:
  Strategy 1: General        — no incident information
  Strategy 2: +Binary        — incident existence flag (is_incident ∈ {0, 1})
  Strategy 3: +TypeEmb       — incident type learnable embedding (9 categories)

Architecture:
  Standard Transformer Encoder (no decoder — we use the mean-pooled output):
    - Learnable positional embeddings for 24 timesteps
    - N encoder layers with multi-head self-attention and feed-forward
    - Mean pooling over time → Linear decoder → 6 outputs.

  Incident information is concatenated to each token's feature vector,
  keeping the per-timestep structure identical to the LSTM approach.

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

GENERAL_FEAT_IDX = [0, 1, 2, 3, 4, 8, 9, 10, 13, 14]
NUM_GENERAL_FEAT = len(GENERAL_FEAT_IDX)   # 10

NUM_INCIDENT_TYPES = 9
TYPE_EMB_DIM       = 16

# Transformer architecture
D_MODEL      = 128      # token embedding dimension
NHEAD        = 8        # number of attention heads
NUM_LAYERS   = 3        # encoder layers
DIM_FEEDFORWARD = 512   # feed-forward hidden dim
DROPOUT      = 0.1
POOLING      = "mean"   # "mean" or "cls"

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
        x_gen_seq = self.input_3d[idx][:, GENERAL_FEAT_IDX].astype(np.float32)
        is_incident = float(self.impact_seq[idx] >= 0.0)
        type_idx    = self.inc_type[idx]
        y           = self.targets[idx].astype(np.float32)

        return {
            "x_gen_seq": torch.from_numpy(x_gen_seq),       # (24, 10)
            "is_inc":    torch.tensor(is_incident, dtype=torch.float32),
            "x_type_idx":torch.tensor(type_idx, dtype=torch.long),
            "y":         torch.from_numpy(y),
        }


# ============================================================================
# 2. Transformer Model
# ============================================================================

class PositionalEncoding(nn.Module):
    """Learnable positional embeddings for 24 timesteps."""
    def __init__(self, d_model, max_len=24, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x: (B, T, D)
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerForecaster(nn.Module):
    """
    Standard Transformer encoder.
    Per-timestep token is projected from (general_feat_dim + incident_info)
    to d_model via a linear layer, then processed by multi-head self-attention.
    """

    def __init__(self, strategy="general"):
        super().__init__()
        self.strategy = strategy.lower()

        self.gen_dim    = NUM_GENERAL_FEAT
        self.binary_dim = NUM_GENERAL_FEAT + 1
        self.type_dim   = NUM_GENERAL_FEAT + TYPE_EMB_DIM

        if self.strategy == "typeemb":
            self.type_emb = nn.Embedding(NUM_INCIDENT_TYPES,
                                         TYPE_EMB_DIM, padding_idx=0)
            token_dim = self.type_dim
        elif self.strategy == "binary":
            token_dim = self.binary_dim
        else:
            token_dim = self.gen_dim

        # Project raw feature vector to d_model
        self.input_proj = nn.Linear(token_dim, D_MODEL)

        # Positional encoding
        self.pos_enc = PositionalEncoding(D_MODEL, max_len=NUM_TIMESTEPS,
                                          dropout=DROPOUT)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, NUM_TARGETS),
        )

        self.pooling = POOLING

    def forward(self, batch):
        x_gen_seq = batch["x_gen_seq"].to(DEVICE)          # (B, 24, 10)
        B, T, _ = x_gen_seq.shape

        # Build per-timestep token
        if self.strategy == "general":
            tokens = x_gen_seq

        elif self.strategy == "binary":
            is_inc = batch["is_inc"].to(DEVICE).unsqueeze(-1).unsqueeze(-1)
            flag = is_inc.expand(B, T, 1)
            tokens = torch.cat([x_gen_seq, flag], dim=-1)

        elif self.strategy == "typeemb":
            type_idx = batch["x_type_idx"].to(DEVICE)
            is_inc   = batch["is_inc"].to(DEVICE)
            emb = self.type_emb(type_idx)                          # (B, 16)
            emb = emb * is_inc.unsqueeze(-1)
            emb = emb.unsqueeze(1).expand(B, T, TYPE_EMB_DIM)
            tokens = torch.cat([x_gen_seq, emb], dim=-1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Project + positional encoding
        x = self.input_proj(tokens)                                # (B, 24, D)
        x = self.pos_enc(x)
        x = self.encoder(x)                                        # (B, 24, D)

        # Pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)                                      # (B, D)
        else:
            x = x[:, 0, :]                                         # CLS token

        return self.decoder(x)


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

    model     = TransformerForecaster(strategy).to(DEVICE)
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
        description="Transformer Model for Post-Incident Traffic Forecasting")
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
                "mae_1h": test_mae_1h, "mae_3h": test_mae_3h, "mae_6h": test_mae_6h,
                "mae_avg": test_mae_avg,
                "mape_1h": test_mape_1h, "mape_3h": test_mape_3h, "mape_6h": test_mape_6h,
                "mape_avg": np.mean([test_mape_1h, test_mape_3h, test_mape_6h]),
                "rmse_1h": test_rmse_1h, "rmse_3h": test_rmse_3h, "rmse_6h": test_rmse_6h,
                "rmse_avg": np.mean([test_rmse_1h, test_rmse_3h, test_rmse_6h]),
            }
        }

    print(f"\n{'='*60}")
    print("FINAL RESULTS — Transformer Model")
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
    with open(save_dir / "transformer_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
