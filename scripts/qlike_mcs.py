"""Complementary MCS on the QLIKE covariance-forecast loss (not FZ0).

The FZ0/MCS comparison is the primary, risk-measure-consistent ranking, but it
barely discriminates: all models share the FHS quantile machinery and differ
only through their conditional volatility, and the FZ0 loss is dominated by the
~1% exceedance days. QLIKE is a strictly consistent loss for the *variance /
covariance* forecast itself (Patton 2011), uses *every* day, and bypasses the
FHS construction -> far more power to separate the underlying Sigma_t forecasts.

Complementary lens ("which model forecasts the conditional covariance best?"),
NOT a replacement for FZ0 ("which forecasts VaR/ES best?").

Two scores per day t (lower = better):
  - portfolio QLIKE:  L_t = log(h_{p,t}) + r_{p,t}^2 / h_{p,t},  h_{p,t}=w'Sigma_t w
  - matrix  QLIKE:    L_t = log|Sigma_t| + r_t' Sigma_t^{-1} r_t   (Stein/QLIKE)

Dataset-agnostic: set QLIKE_TAG to the as-of/cohort tag. Assets and N are read
from the split CSV header, so it works for the 4-asset main set and the 2-asset
robustness cohorts alike.
    QLIKE_TAG=2026-05-03      python scripts/qlike_mcs.py   # main 4-asset
    QLIKE_TAG=gspc_tnx_1990   python scripts/qlike_mcs.py   # 2-asset, 1990
    QLIKE_TAG=gspc_tnx_1962   python scripts/qlike_mcs.py   # 2-asset, full history

Reconstructs Sigma_t from saved artifacts (neural checkpoints + R BEKK/DCC
outputs), no retraining. Output -> results/evaluation_qlike/<TAG>/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arch.bootstrap import MCS
from vech import vech
from model_names import is_legacy_alias, MODEL_LABELS
from make_fhs_var_allmodels import build_model  # dataset-agnostic constructor

TAG = os.environ.get("QLIKE_TAG", "2026-05-03")
LOOKBACK = 40
JITTER = 1e-12
MCS_SIZE, MCS_REPS, MCS_SEED = 0.10, 1000, 42

DATA = ROOT / "data"
RUNS = ROOT / "results" / "runs"
ROUT = ROOT / "r" / "output"
OUT = ROOT / "results" / "evaluation_qlike" / TAG

R_MODELS = {
    "bekk_symmetric": ("bekk", "symmetric"),
    "bekk_asymmetric": ("bekk", "asymmetric"),
    "dcc": ("dcc", "dcc"),
    "adcc": ("dcc", "adcc"),
}

# Assets / N are read from the split header -> works for N=2 and N=4.
ASSETS = [c for c in pd.read_csv(DATA / f"train_df_{TAG}.csv", nrows=0).columns
          if c != "Date"]
N = len(ASSETS)
W = np.full(N, 1.0 / N)
HCOLS = [f"h_{i}{j}" for i in range(1, N + 1) for j in range(1, N + 1)]


# --------------------------------------------------------------------------- #
# Data + Sigma_t reconstruction (parameterized by ASSETS / TAG)
# --------------------------------------------------------------------------- #
def add_cross_returns(df: pd.DataFrame) -> pd.DataFrame:
    R = df[ASSETS].to_numpy(dtype=np.float32)
    U = R[:, :, None] * R[:, None, :]
    cross = vech(torch.from_numpy(U)).numpy()
    cols = [f"cross_{i + 1}" for i in range(cross.shape[1])]
    out = df[ASSETS].copy()
    out[cols] = cross
    return out


def load_splits():
    raw = {s: pd.read_csv(DATA / f"{s}_df_{TAG}.csv", parse_dates=["Date"]).set_index("Date")
           for s in ("train", "val", "test")}
    feats = {s: add_cross_returns(df) for s, df in raw.items()}
    mu, sigma = feats["train"].mean(), feats["train"].std()
    norm = {s: (df - mu) / sigma for s, df in feats.items()}
    train_norm = norm["train"][ASSETS].to_numpy(dtype=np.float32)
    train_cov = torch.from_numpy(np.cov(train_norm.T).astype(np.float32))
    return_std = torch.from_numpy(sigma[ASSETS].to_numpy(dtype=np.float32))
    return raw, norm, sigma, train_cov, return_std


def make_loader(norm_df: pd.DataFrame) -> torch.Tensor:
    X = norm_df.to_numpy(dtype=np.float32)
    seqs = np.stack([X[i:i + LOOKBACK] for i in range(len(X) - LOOKBACK)])
    return torch.from_numpy(seqs)


def neural_sigma(model, test_loader, sigma_std, canonical_dates, test_index) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        sigma_scaled, _ = model(test_loader)
    sigma_scaled = sigma_scaled.cpu().numpy()
    std = sigma_std[ASSETS].to_numpy(dtype=np.float32)
    sigma_real = sigma_scaled * std[None, :, None] * std[None, None, :]
    s = pd.Series(list(sigma_real), index=test_index[LOOKBACK:])
    return np.stack(s.reindex(canonical_dates).to_numpy())


def r_sigma(prefix: str, kind: str, seed: str, canonical_dates) -> np.ndarray:
    te = ROUT / f"{prefix}_forecasts_{kind}_asof_{TAG}_seed{seed}.csv"
    df = pd.read_csv(te, parse_dates=["Date"]).set_index("Date")[HCOLS]
    return df.reindex(canonical_dates).to_numpy().reshape(-1, N, N)


# --------------------------------------------------------------------------- #
def portfolio_qlike(Sigma: np.ndarray, r_assets: np.ndarray) -> np.ndarray:
    h = np.clip(np.einsum("i,tij,j->t", W, Sigma, W), 1e-24, None)
    rp2 = (r_assets @ W) ** 2
    return np.log(h) + rp2 / h


def matrix_qlike(Sigma: np.ndarray, r_assets: np.ndarray) -> np.ndarray:
    eye = np.eye(N) * JITTER
    out = np.empty(len(Sigma))
    for t, (S, r) in enumerate(zip(Sigma, r_assets)):
        Sj = S + eye
        L = np.linalg.cholesky(Sj)
        out[t] = 2.0 * np.log(np.diag(L)).sum() + r @ np.linalg.solve(Sj, r)
    return out


def run_mcs(loss_df: pd.DataFrame) -> tuple[set[str], pd.Series]:
    mcs = MCS(loss_df, size=MCS_SIZE, reps=MCS_REPS, bootstrap="stationary", seed=MCS_SEED)
    mcs.compute()
    return set(mcs.included), mcs.pvalues["Pvalue"]


def main() -> None:
    print(f"QLIKE-MCS on TAG={TAG}: assets={ASSETS} (N={N})")
    raw, norm, sigma_std, train_cov, return_std = load_splits()
    loaders = {s: make_loader(norm[s]) for s in norm}
    canonical_dates = raw["test"].index[LOOKBACK:]
    r_assets = raw["test"].loc[canonical_dates, ASSETS].to_numpy(dtype=float)

    seed_dirs = sorted(RUNS.glob(f"asof_{TAG}_seed*"))
    if not seed_dirs:
        raise SystemExit(f"no run dirs under {RUNS}/asof_{TAG}_seed*")

    rows = {"portfolio": [], "matrix": []}
    for d in seed_dirs:
        seed = d.name.split("seed")[-1]
        sigmas: dict[str, np.ndarray] = {}
        for name in sorted(p.name for p in (d / "models").glob("*")
                           if p.is_dir() and not is_legacy_alias(p.name)):
            mdir = d / "models" / name
            ckpt, cfg = mdir / "checkpoint.pt", mdir / "config.json"
            if not (ckpt.exists() and cfg.exists()):
                continue
            try:
                config = json.load(open(cfg))
                model = build_model(config, train_cov, return_std)
                model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                                 weights_only=False)["model_state_dict"])
                sigmas[name] = neural_sigma(model, loaders["test"], sigma_std,
                                            canonical_dates, raw["test"].index)
            except Exception as exc:
                print(f"  [skip] {name} seed{seed}: {type(exc).__name__}: {exc}")
        for name, (prefix, kind) in R_MODELS.items():
            try:
                sigmas[name] = r_sigma(prefix, kind, seed, canonical_dates)
            except FileNotFoundError:
                pass

        if len(sigmas) < 2:
            print(f"  [skip] seed {seed}: <2 models")
            continue
        for variant, fn in (("portfolio", portfolio_qlike), ("matrix", matrix_qlike)):
            loss_df = pd.DataFrame({m: fn(S, r_assets) for m, S in sigmas.items()})
            loss_df = loss_df.replace([np.inf, -np.inf], np.nan).dropna()
            included, pvals = run_mcs(loss_df)
            for m in loss_df.columns:
                rows[variant].append({"seed": seed, "model": m,
                                      "mean_qlike": float(loss_df[m].mean()),
                                      "in_mcs": m in included,
                                      "pval_mcs": float(pvals.get(m, np.nan))})
        print(f"  seed {seed}: {len(sigmas)} models scored")

    OUT.mkdir(parents=True, exist_ok=True)
    for variant in ("portfolio", "matrix"):
        df = pd.DataFrame(rows[variant])
        n_seeds = df["seed"].nunique()
        agg = (df.groupby("model")
                 .agg(mean_qlike=("mean_qlike", "mean"),
                      in_mcs=("in_mcs", "sum"),
                      pval_mcs=("pval_mcs", "mean"))
                 .sort_values("mean_qlike"))
        agg["in_mcs"] = agg["in_mcs"].astype(int).astype(str) + f"/{n_seeds}"
        agg.index = [MODEL_LABELS.get(m, m) for m in agg.index]
        df.to_csv(OUT / f"qlike_{variant}_per_seed.csv", index=False)
        agg.to_csv(OUT / f"qlike_{variant}_summary.csv")
        print(f"\n=== {variant.upper()} QLIKE  (lower = better, {n_seeds} seeds, TAG={TAG}) ===")
        print(agg.to_string())
    print(f"\nsaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
