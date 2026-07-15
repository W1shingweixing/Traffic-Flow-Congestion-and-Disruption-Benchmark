"""
STGCN — Spatial-Temporal Graph Convolutional Network
=====================================================
Reference: Yu et al. (2018), IJCAI 2018.

Uses graph-specific dataset: graph_train/val/test_24_6.csv
Each row = all 115 stations × (24h history + 6h target).
Feeds (B, C, N, T) to the STGCN blocks.

Supports 3 strategies: General / +Binary / +TypeEmb
Outputs: MAE, MAPE, RMSE, JSON, log file.
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse, json, sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("C:/Users/huang/Desktop/traffic_data_24_6/graph dataset")
TRAIN_PATH = DATA_DIR / "graph_train_24_6.csv"
VAL_PATH   = DATA_DIR / "graph_val_24_6.csv"
TEST_PATH  = DATA_DIR / "graph_test_24_6.csv"
ADJ_PATH   = Path("C:/Users/huang/Desktop/traffic_data_24_6/adjacent matrix/adj_matrix_nsw.npy")

# ── Constants ──────────────────────────────────────────────────────────────
NUM_TIMESTEPS, FEAT_PER_STEP, NUM_TARGETS, NUM_STATIONS = 24, 10, 6, 115
NUM_FEAT_PER_STATION = NUM_TIMESTEPS * FEAT_PER_STEP  # 240
TOTAL_INPUT  = NUM_STATIONS * NUM_FEAT_PER_STATION      # 27600
TOTAL_TARGET = NUM_STATIONS * NUM_TARGETS               # 690

NUM_INCIDENT_TYPES, TYPE_EMB_DIM = 9, 16

CHEB_K, GCN_CH, TCN_CH, TCN_KS, DROPOUT = 3, 64, 64, 3, 0.2
BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR = 8, 200, 15, 1e-3
SEEDS = [42, 123, 777]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scaled_laplacian(W):
    n = W.shape[0]; d = W.sum(axis=1)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6)))
    L_norm = np.eye(n) - D_sqrt_inv @ W @ D_sqrt_inv
    return ((2.0 / 2.0) * L_norm - np.eye(n)).astype(np.float32)


def load_adjacency():
    dist_mat = np.load(ADJ_PATH)
    A = (dist_mat > 0).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    d = A.sum(axis=1)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6)))
    A_norm = (D_sqrt_inv @ A @ D_sqrt_inv + D_sqrt_inv @ A.T @ D_sqrt_inv) / 2.0
    return scaled_laplacian(A_norm)


class GraphWindowDataset(Dataset):
    """Loads graph_train/val/test_24_6.csv. Each row = (27600 inputs, 690 targets)."""
    def __init__(self, csv_path, strategy="general"):
        self.strategy = strategy.lower()
        raw = pd.read_csv(csv_path, header=None, dtype=np.float32).values
        self.X = raw[:, :TOTAL_INPUT]                                      # (N_samples, 27600)
        self.Y = raw[:, TOTAL_INPUT:TOTAL_INPUT + TOTAL_TARGET]            # (N_samples, 690)
        # Extract incident features for each station (impact_sequence_hour is col 5 per station)
        # Reshape to (N_samples, N_stations, 240) then extract col 5 of first timestep
        self.X_3d = self.X.reshape(-1, NUM_STATIONS, NUM_TIMESTEPS, FEAT_PER_STEP)
        self.n_samples  = self.X.shape[0]
        self.impact_seq = self.X_3d[:, :, 0, 5].copy()                   # col 5=impact_seq, t=0
        self.inc_type   = np.zeros((self.n_samples, NUM_STATIONS), dtype=np.int64)
        self.n_samples  = self.X.shape[0]

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        # Shape: (N_stations, T_in, C) for each sample
        x = self.X_3d[idx].astype(np.float32)               # (115, 24, 10)
        y = self.Y[idx].reshape(NUM_STATIONS, NUM_TARGETS)  # (115, 6)
        y = y.astype(np.float32)
        return {
            "x_stations": torch.from_numpy(x),
            "is_inc":     torch.from_numpy((self.impact_seq[idx] >= 0).astype(np.float32)),
            "x_type":     torch.from_numpy(self.inc_type[idx]).long(),
            "y":          torch.from_numpy(y),
        }


# ── STGCN blocks (batched) ─────────────────────────────────────────────────

class TemporalGatedConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, (1, kernel_size), padding=(0, kernel_size - 1))
    def forward(self, x):
        P, Q = self.conv(x).chunk(2, dim=1)
        return (P + Q) * torch.sigmoid(P - Q)


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, K):
        super().__init__()
        self.K = K
        self.W = nn.Parameter(torch.randn(K, c_in, c_out) * 0.02)
        self.B = nn.Parameter(torch.zeros(c_out))
    def forward(self, x, L):
        B, C, N = x.shape; L = L.to(x.device).unsqueeze(0).expand(B, N, N)
        x = x.permute(0, 2, 1); T0 = x; out = T0 @ self.W[0]
        if self.K > 1: T1 = torch.bmm(L, T0); out = out + T1 @ self.W[1]
        else: T1 = None
        Tkm2, Tkm1 = T0, T1
        for k in range(2, self.K):
            Tk = 2 * torch.bmm(L, Tkm1) - Tkm2; out = out + Tk @ self.W[k]; Tkm2, Tkm1 = Tkm1, Tk
        return out.permute(0, 2, 1) + self.B.view(1, -1, 1)


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_gcn, c_tcn, K, kt):
        super().__init__()
        self.tcn1 = TemporalGatedConv(c_in, c_tcn, kt)
        self.gcn  = ChebGraphConv(c_tcn, c_gcn, K)
        self.tcn2 = TemporalGatedConv(c_gcn, c_tcn, kt)
        self.bn   = nn.BatchNorm2d(c_tcn)
    def forward(self, x, L):
        x = self.tcn1(x); x = x[:,:,:,:NUM_TIMESTEPS]; x = self.bn(x)
        B, C, N, T = x.shape
        gcn_out = []
        for ti in range(T):
            gcn_out.append(self.gcn(x[:,:,:,ti], L).unsqueeze(-1))
        x = torch.cat(gcn_out, -1)
        x = self.tcn2(x); return x[:,:,:,:NUM_TIMESTEPS]


class STGCNForecaster(nn.Module):
    def __init__(self, strategy="general"):
        super().__init__()
        self.strategy = strategy.lower()
        gd, bd, td = FEAT_PER_STEP, FEAT_PER_STEP + 1, FEAT_PER_STEP + TYPE_EMB_DIM
        if self.strategy == "typeemb":
            self.te = nn.Embedding(NUM_INCIDENT_TYPES, TYPE_EMB_DIM, padding_idx=0)
            ci = td
        elif self.strategy == "binary": ci = bd
        else: ci = gd
        self.b1 = STGCNBlock(ci, GCN_CH, TCN_CH, CHEB_K, TCN_KS)
        self.b2 = STGCNBlock(TCN_CH, GCN_CH, TCN_CH, CHEB_K, TCN_KS)
        self.fc = nn.Sequential(nn.Linear(TCN_CH * NUM_TIMESTEPS * NUM_STATIONS, 256),
                                nn.ReLU(), nn.Dropout(DROPOUT),
                                nn.Linear(256, NUM_STATIONS * NUM_TARGETS))
        self.register_buffer("L", torch.from_numpy(load_adjacency()))

    def forward(self, batch):
        # x_stations: (B, N, T, C) → permute to (B, C, N, T)
        x = batch["x_stations"].to(DEVICE).permute(0, 3, 1, 2)
        B, C, N, T = x.shape
        if self.strategy == "binary":
            f = batch["is_inc"].to(DEVICE).unsqueeze(1).unsqueeze(-1).expand(B, 1, N, T)
            x = torch.cat([x, f], 1)
        elif self.strategy == "typeemb":
            e = self.te(batch["x_type"].to(DEVICE)) * batch["is_inc"].to(DEVICE).unsqueeze(-1)
            e = e.permute(0, 2, 1).unsqueeze(-1).expand(B, TYPE_EMB_DIM, N, T)
            x = torch.cat([x, e], 1)
        x = self.b1(x, self.L)
        x = self.b2(x, self.L)                              # (B, TCN_CH, N, T)
        x = x.permute(0, 2, 1, 3).reshape(B, N * TCN_CH * T)
        return self.fc(x).view(B, NUM_STATIONS, NUM_TARGETS)


# ── Training ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, crit):
    model.train(); total = 0.0
    for b in loader:
        p = model(b); loss = crit(p, b["y"].to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(b["y"])
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P, T = [], []
    for b in loader:
        P.append(model(b).cpu().numpy()); T.append(b["y"].cpu().numpy())
    p = np.concatenate(P, 0).reshape(-1, NUM_TARGETS)
    t = np.concatenate(T, 0).reshape(-1, NUM_TARGETS)
    mae = np.mean(np.abs(p - t), 0); rmse = np.sqrt(np.mean((p - t) ** 2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((t - p) / (t + 1e-8)); ape = np.where(t == 0, 0, ape)
        mape = np.mean(ape, 0) * 100
    return {
        "mae_1h": float(mae[0]), "mape_1h": float(mape[0]), "rmse_1h": float(rmse[0]),
        "mae_3h": float(mae[2]), "mape_3h": float(mape[2]), "rmse_3h": float(rmse[2]),
        "mae_6h": float(mae[5]), "mape_6h": float(mape[5]), "rmse_6h": float(rmse[5]),
        "mae_avg": float(np.mean(mae)), "mape_avg": float(np.mean(mape)),
        "rmse_avg": float(np.mean(rmse)),
    }


def run(strategy, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(GraphWindowDataset(TRAIN_PATH, strategy), BATCH_SIZE, True)
    va = DataLoader(GraphWindowDataset(VAL_PATH,   strategy), BATCH_SIZE)
    te = DataLoader(GraphWindowDataset(TEST_PATH,  strategy), BATCH_SIZE)
    model = STGCNForecaster(strategy).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.L1Loss()
    best_mae, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        train_epoch(model, tr, opt, crit); vm = evaluate(model, va)
        if ep % 2 == 1 or ep <= 3:
            print(f"    Epoch {ep:3d} | Val MAE {vm['mae_avg']:.4f} (best {best_mae:.4f})")
        if vm["mae_avg"] < best_mae:
            best_mae = vm["mae_avg"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val = vm.copy(); no_imp = 0
        else: no_imp += 1
        if no_imp >= PATIENCE: break
    model.load_state_dict(best_state)
    return {"seed": seed, "train": evaluate(model, tr),
            "val": best_val, "test": evaluate(model, te)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=["general","binary","typeemb","all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--save_dir", default="C:/Users/huang/Desktop/cloude code/results")
    args = parser.parse_args()
    sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
    strategies = ["general","binary","typeemb"] if args.strategy == "all" else [args.strategy]
    R = {}
    for s in strategies:
        print(f"\n{'='*60}\nRunning Strategy: {s.upper()}\n{'='*60}")
        sr = [run(s, se) for se in args.seeds]
        for r in sr:
            print(f"  Seed {r['seed']}: Val MAE={r['val']['mae_avg']:.4f}  Test MAE={r['test']['mae_avg']:.4f}")
        t = lambda k: np.mean([r["test"][k] for r in sr])
        R[s] = {"individual": sr, "summary": {
            "mae_1h": t("mae_1h"),"mae_3h": t("mae_3h"),"mae_6h": t("mae_6h"),"mae_avg": t("mae_avg"),
            "mape_1h": t("mape_1h"),"mape_3h": t("mape_3h"),"mape_6h": t("mape_6h"),
            "mape_avg": np.mean([t(k) for k in ["mape_1h","mape_3h","mape_6h"]]),
            "rmse_1h": t("rmse_1h"),"rmse_3h": t("rmse_3h"),"rmse_6h": t("rmse_6h"),
            "rmse_avg": np.mean([t(k) for k in ["rmse_1h","rmse_3h","rmse_6h"]]),
        }}
    print(f"\n{'='*60}\nFINAL RESULTS — STGCN Model\n{'='*60}")
    h = f"{'Strategy':<12} {'MAE-1h':>8} {'MAE-3h':>8} {'MAE-6h':>8} {'MAE-avg':>8} {'MAPE-1h':>8} {'MAPE-3h':>8} {'MAPE-6h':>8} {'RMSE-1h':>8} {'RMSE-3h':>8} {'RMSE-6h':>8}"
    print(h); print("-"*len(h))
    for s in ["general","binary","typeemb"]:
        if s not in R: continue
        d = R[s]["summary"]
        print(f"{s:<12} {d['mae_1h']:8.4f} {d['mae_3h']:8.4f} {d['mae_6h']:8.4f} {d['mae_avg']:8.4f} {d['mape_1h']:7.2f}% {d['mape_3h']:7.2f}% {d['mape_6h']:7.2f}% {d['rmse_1h']:8.4f} {d['rmse_3h']:8.4f} {d['rmse_6h']:8.4f}")
    if "general" in R and "binary" in R:
        print("\nAblation Deltas:")
        for hh in["1h","3h","6h"]:
            g=R["general"]["summary"][f"mae_{hh}"]; b=R["binary"]["summary"][f"mae_{hh}"]
            print(f"  D1 (Binary - General) {hh}: {b-g:+.4f}")
        if "typeemb" in R:
            for hh in["1h","3h","6h"]:
                b=R["binary"]["summary"][f"mae_{hh}"]; t_=R["typeemb"]["summary"][f"mae_{hh}"]
                print(f"  D2 (TypeEmb - Binary) {hh}: {t_-b:+.4f}")
    with open(sd/"stgcn_results.json","w") as f: json.dump(R,f,indent=2,default=float)


if __name__ == "__main__": main()
