"""
STSGCN — Spatial-Temporal Synchronous Graph Convolutional Network
==================================================================
Reference: Song et al. (2020), AAAI 2020.

Uses graph-specific dataset: graph_train/val/test_24_6.csv
Each row = all 115 stations × (24h history + 6h target).

Key parameters: T=2, EMBED_DIM=64, N_CONV=2, N_CL=2, DROPOUT=0.2.
Supports 3 strategies: General / +Binary / +TypeEmb.
Outputs: MAE, MAPE, RMSE, JSON, log file.
"""

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse, json, sys
from pathlib import Path

DATA_DIR   = Path("C:/Users/huang/Desktop/traffic_data_24_6/graph dataset")
TRAIN_PATH = DATA_DIR / "graph_train_24_6.csv"
VAL_PATH   = DATA_DIR / "graph_val_24_6.csv"
TEST_PATH  = DATA_DIR / "graph_test_24_6.csv"
ADJ_MATRIX_DIR = Path("C:/Users/huang/Desktop/traffic_data_24_6/adjacent matrix")

NUM_TIMESTEPS, FEAT_PER_STEP, NUM_TARGETS, NUM_STATIONS = 24, 10, 6, 115
NUM_FEAT_PER_STATION = NUM_TIMESTEPS * FEAT_PER_STEP   # 240
TOTAL_INPUT  = NUM_STATIONS * NUM_FEAT_PER_STATION       # 27600
TOTAL_TARGET = NUM_STATIONS * NUM_TARGETS                # 690

NUM_INCIDENT_TYPES, TYPE_EMB_DIM = 9, 16

T_DIM, EMBED_DIM, MASK_INIT, N_CONV, N_CL, DROPOUT = 2, 64, 0.5, 2, 2, 0.2
BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR = 8, 200, 15, 1e-3
SEEDS = [42, 123, 777]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_components(dist_mat, N, Td, ed, mask_init):
    non_zero = dist_mat > 0
    sigma = dist_mat[non_zero].std() if non_zero.any() else 10.0
    W = np.where(non_zero, np.exp(-dist_mat ** 2 / sigma ** 2), 0.0).astype(np.float32)

    L = np.zeros((Td * N, Td * N), dtype=np.float32)
    M = np.zeros((Td * N, Td * N), dtype=np.float32)
    E = np.zeros((Td * N, ed), dtype=np.float32)

    for t in range(Td):
        L[t*N:(t+1)*N, t*N:(t+1)*N] = W
        M[t*N:(t+1)*N, t*N:(t+1)*N] = W * mask_init
        if t < Td - 1:
            L[t*N:(t+1)*N, (t+1)*N:(t+2)*N] = np.eye(N, dtype=np.float32)
            L[(t+1)*N:(t+2)*N, t*N:(t+1)*N] = np.eye(N, dtype=np.float32)
            M[t*N:(t+1)*N, (t+1)*N:(t+2)*N] = np.eye(N, dtype=np.float32) * mask_init

    for t in range(Td):
        for n in range(N):
            idx = t * N + n; E[idx, t] = 1.0
            for d in range(Td, ed):
                E[idx, d] = np.sin(n / (10000 ** (d / ed))) if d % 2 == 0 else np.cos(n / (10000 ** (d / ed)))
    return L, M, E


class GraphWindowDataset(Dataset):
    def __init__(self, csv_path, strategy="general"):
        self.strategy = strategy.lower()
        raw = pd.read_csv(csv_path, header=None, dtype=np.float32).values
        self.X = raw[:, :TOTAL_INPUT]
        self.Y = raw[:, TOTAL_INPUT:TOTAL_INPUT + TOTAL_TARGET]
        self.X_3d = self.X.reshape(-1, NUM_STATIONS, NUM_TIMESTEPS, FEAT_PER_STEP)
        self.n_samples = self.X.shape[0]
        self.impact_seq = self.X_3d[:, :, 0, 5].copy()
        self.inc_type   = np.zeros((self.n_samples, NUM_STATIONS), dtype=np.int64)
        self.n_samples = self.X.shape[0]

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        x = self.X_3d[idx].astype(np.float32)                                       # (115, 24, 10)
        y = self.Y[idx].reshape(NUM_STATIONS, NUM_TARGETS).astype(np.float32)       # (115, 6)
        return {
            "x_stations": torch.from_numpy(x),
            "is_inc":     torch.from_numpy((self.impact_seq[idx] >= 0).astype(np.float32)),
            "x_type":     torch.from_numpy(self.inc_type[idx]).long(),
            "y":          torch.from_numpy(y),
        }


class GraphConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.w1 = nn.Linear(ci, co); self.w2 = nn.Linear(ci, co)
        self.b1 = nn.Parameter(torch.zeros(co)); self.b2 = nn.Parameter(torch.zeros(co))
    def forward(self, x, A):
        g = A.to(x.device) @ x
        return (self.w2(g) + self.b2) * torch.sigmoid(self.w1(g) + self.b1)


class STSGCM(nn.Module):
    def __init__(self, ci, co, nl, dp):
        super().__init__()
        self.gc = nn.ModuleList([GraphConv(ci, co)] + [GraphConv(co, co) for _ in range(1, nl)])
        self.dp = nn.Dropout(dp)
    def forward(self, x, A):
        o = []
        for g in self.gc:
            x = g(x, A); x = self.dp(x); o.append(x)
        return torch.stack(o).max(0)[0]


class STSGCL(nn.Module):
    def __init__(self, Td, N, ed, ngc, ncm, dp):
        super().__init__()
        self.T = Td; self.N = N
        self.stsgcm = nn.ModuleList([STSGCM(ed, ed, ngc, dp) for _ in range(ncm)])
    def forward(self, x, AL, st_emb, mask_mat):
        B, Ti, N, feat = x.shape; T = self.T
        A_adj = AL.to(x.device) * mask_mat.to(x.device)
        emb = st_emb.to(x.device)
        outs = []
        for t in range(Ti - T + 1):
            win = x[:, t:t+T, :, :].permute(0, 2, 1, 3).reshape(B, T * N, feat)
            win = win + emb.unsqueeze(0)
            out = self.stsgcm[t](win, A_adj)
            outs.append(out[:, (T//2)*N:(T//2+1)*N, :])
        return torch.stack(outs, 1)


class STSGCNForecaster(nn.Module):
    def __init__(self, strategy="general"):
        super().__init__()
        self.strategy = strategy.lower()
        self.T = T_DIM; self.N = NUM_STATIONS
        gd, bd, td = FEAT_PER_STEP, FEAT_PER_STEP + 1, FEAT_PER_STEP + TYPE_EMB_DIM
        if self.strategy == "typeemb":
            self.te = nn.Embedding(NUM_INCIDENT_TYPES, TYPE_EMB_DIM, padding_idx=0)
            ci = td
        elif self.strategy == "binary": ci = bd
        else: ci = gd
        self.inp = nn.Linear(ci, EMBED_DIM)
        # Load pre-built STSGCN components from adjacent matrix directory
        self.register_buffer("AL", torch.from_numpy(np.load(ADJ_MATRIX_DIR / "stsgcn_local_stg_T2.npy").astype(np.float32)))
        self.MM = nn.Parameter(torch.from_numpy(np.load(ADJ_MATRIX_DIR / "stsgcn_mask_T2.npy").astype(np.float32)))
        self.register_buffer("SE", torch.from_numpy(np.load(ADJ_MATRIX_DIR / "stsgcn_st_embedding_T2.npy").astype(np.float32)))
        ncm = NUM_TIMESTEPS - T_DIM + 1
        self.cl1 = STSGCL(T_DIM, NUM_STATIONS, EMBED_DIM, N_CONV, ncm, DROPOUT)
        self.cl2 = STSGCL(T_DIM, NUM_STATIONS, EMBED_DIM, N_CONV, ncm - T_DIM + 1, DROPOUT)
        self.fl = NUM_TIMESTEPS - 2 * T_DIM + 2
        self.dec = nn.Sequential(
            nn.Linear(EMBED_DIM * self.fl * NUM_STATIONS, 512),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_STATIONS * NUM_TARGETS),
        )

    def forward(self, batch):
        x = batch["x_stations"].to(DEVICE)                                         # (B, N, T, C)
        B, N, Ti, C = x.shape
        if self.strategy == "binary":
            f = batch["is_inc"].to(DEVICE).unsqueeze(-1).unsqueeze(-1).expand(B, N, Ti, 1)
            x = torch.cat([x, f], -1)
        elif self.strategy == "typeemb":
            e = self.te(batch["x_type"].to(DEVICE)) * batch["is_inc"].to(DEVICE).unsqueeze(-1)
            x = torch.cat([x, e.unsqueeze(2).expand(B, N, Ti, TYPE_EMB_DIM)], -1)
        x = self.inp(x).permute(0, 2, 1, 3)                                        # (B, Ti, N, ed)
        x = self.cl1(x, self.AL, self.SE, self.MM)
        x = self.cl2(x, self.AL, self.SE, self.MM)                                 # (B, Ti_out, N, ed)
        x = x.permute(0, 2, 1, 3).reshape(B, N, -1)                                # (B, N, Ti_out*ed)
        return self.dec(x.reshape(B, -1)).view(B, NUM_STATIONS, NUM_TARGETS)


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
    model = STSGCNForecaster(strategy).to(DEVICE)
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
    print(f"\n{'='*60}\nFINAL RESULTS — STSGCN Model\n{'='*60}")
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
    with open(sd/"stsgcn_results.json","w") as f: json.dump(R,f,indent=2,default=float)


if __name__ == "__main__": main()
