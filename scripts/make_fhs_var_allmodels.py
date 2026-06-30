"""Generate one-step-ahead FHS VaR/ES figures for every trained neural model.

For each model that was trained across the seed runs
(``results/runs/asof_<ASOF>_seed*/models/<model>/``) this script:

  1. rebuilds the exact training inputs offline from the saved return splits
     (4 asset returns + 10 cross-returns = 14 features, train-normalized),
  2. reloads the fixed seed-42 checkpoint and runs a forward pass over each split to
     obtain the conditional covariance Sigma_t,
  3. forms the equal-weight portfolio and runs filtered historical simulation
     (FHS) for VaR/ES,
  4. renders each model with the canonical thesis crisis shading into
     ``figures/vares_plots/fig_fhs_var_<model>.pdf``.

FHS is invariant to a constant rescaling of volatility, and the only scale that
differs between models is constant, so all model families are treated uniformly
here (no per-distribution special casing).

Run from the repo root:  python scripts/make_fhs_var_allmodels.py
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vech import vech
from risk_metrics import fhs_var_es
from plots import plot_fhs_var
from LSTMCovariance import LSTMCovariance
from GRUCovariance import GRUCovariance
from bekk_kernel import BEKKLSTM
from neural_bekk import NeuralBekk
from model_names import model_label, is_legacy_alias

ASOF = "2026-05-03"
ALPHA = 0.01
WINDOW = 1000
LOOKBACK = 40
ASSETS = ["^GSPC", "GC=F", "CL=F", "^TNX"]
N = len(ASSETS)
W = np.full(N, 1.0 / N, dtype=np.float32)  # equal-weight portfolio
DEVICE = "cpu"
FIGURE_SEED = 42

DATA = ROOT / "data"
RUNS = ROOT / "results" / "runs"
OUTDIR = ROOT / "figures" / "vares_plots"

MODEL_CLASSES = {
    "LSTMCovariance": LSTMCovariance,
    "GRUCovariance": GRUCovariance,
    "BEKKLSTM": BEKKLSTM,
    "NeuralBekk": NeuralBekk,
}

# Figure-header titles come from the canonical labels in model_names.py.


# --------------------------------------------------------------------------- #
# Data (identical across seeds -> built once)
# --------------------------------------------------------------------------- #
def add_cross_returns(df_returns: pd.DataFrame) -> pd.DataFrame:
    """Return df with the 4 asset returns + 10 cross-returns (14 features).

    Cross-returns are vech(R_t R_t^T); they depend only on the contemporaneous
    return vector, so computing them per split equals computing them on the
    full series and slicing.
    """
    R = df_returns[ASSETS].to_numpy(dtype=np.float32)
    U = R[:, :, None] * R[:, None, :]                     # (T, d, d)
    cross = vech(torch.from_numpy(U)).numpy()             # (T, d(d+1)/2)
    cross_cols = [f"cross_{i + 1}" for i in range(cross.shape[1])]
    out = df_returns[ASSETS].copy()
    out[cross_cols] = cross
    return out


def load_splits():
    """Return dict split -> (real_returns_df, normalized_features_df) plus the
    train sample covariance and per-asset std used for rescaling."""
    raw = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(DATA / f"{split}_df_{ASOF}.csv", parse_dates=["Date"]).set_index("Date")
        raw[split] = df

    feats = {split: add_cross_returns(df) for split, df in raw.items()}
    mu = feats["train"].mean()
    sigma = feats["train"].std()
    norm = {split: (df - mu) / sigma for split, df in feats.items()}

    train_norm_returns = norm["train"][ASSETS].to_numpy(dtype=np.float32)
    train_cov = torch.from_numpy(np.cov(train_norm_returns.T).astype(np.float32))
    return_std = torch.from_numpy(sigma[ASSETS].to_numpy(dtype=np.float32))
    return raw, norm, sigma, train_cov, return_std


def make_loader(norm_df: pd.DataFrame):
    """One sample per lookback window; no shuffle, single big batch is fine."""
    X = norm_df.to_numpy(dtype=np.float32)
    seqs = np.stack([X[i:i + LOOKBACK] for i in range(len(X) - LOOKBACK)])
    return torch.from_numpy(seqs)


# --------------------------------------------------------------------------- #
# Model forward + FHS
# --------------------------------------------------------------------------- #
def build_model(config, train_cov, return_std):
    cls = MODEL_CLASSES[config["model_class"]]
    valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in config.items() if k in valid}
    if "Sigma0" in valid:
        kwargs["Sigma0"] = train_cov
    if "return_std" in valid:
        kwargs["return_std"] = return_std
    return cls(**kwargs)


def predict_sigma_real(model, seq_tensor, sigma_std):
    """Forward pass -> Sigma_t on the original return scale."""
    model.eval()
    with torch.no_grad():
        sigma_scaled, _ = model(seq_tensor.to(DEVICE))
    sigma_scaled = sigma_scaled.cpu().numpy()
    std = sigma_std[ASSETS].to_numpy(dtype=np.float32)
    return sigma_scaled * std[None, :, None] * std[None, None, :]


def portfolio(real_df, sigma_real):
    idx = real_df.index[LOOKBACK:]
    r = real_df.loc[idx, ASSETS].to_numpy(dtype=np.float32)
    r_p = r @ W
    var_p = np.einsum("i,tij,j->t", W, sigma_real, W)
    vol_p = np.sqrt(np.clip(var_p, 1e-24, None))
    return idx, r_p, vol_p


def run_one(model, loaders, real, sigma_std):
    """Return (test_dates, r_test, var, es, hits, hit_rate) for one checkpoint."""
    z_parts = []
    test_pack = None
    for split in ("train", "val", "test"):
        sigma_real = predict_sigma_real(model, loaders[split], sigma_std)
        idx, r_p, vol_p = portfolio(real[split], sigma_real)
        if split in ("train", "val"):
            z_parts.append(r_p / np.clip(vol_p, 1e-12, None))
        if split == "test":
            test_pack = (idx, r_p, vol_p)

    z_hist = np.concatenate(z_parts)
    idx, r_p, vol_p = test_pack
    var, es, hits = fhs_var_es(
        z_hist_init=z_hist, r_oos=r_p, vol_oos=vol_p, alpha=ALPHA, window=WINDOW
    )
    return idx, r_p, var, es, hits, float(hits.mean())


# --------------------------------------------------------------------------- #
def main():
    raw, norm, sigma_std, train_cov, return_std = load_splits()
    loaders = {split: make_loader(norm[split]) for split in norm}

    seed_dirs = [RUNS / f"asof_{ASOF}_seed{FIGURE_SEED}"]
    if not seed_dirs[0].is_dir():
        raise SystemExit(f"no fixed-seed run directory: {seed_dirs[0]}")

    # all model names present in any seed (skip stale legacy-alias duplicate dirs)
    model_names = sorted({p.name for d in seed_dirs for p in (d / "models").glob("*")
                          if p.is_dir() and not is_legacy_alias(p.name)})
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"{len(seed_dirs)} seeds, {len(model_names)} models -> {OUTDIR}\n")
    summary = []
    for name in model_names:
        candidates = []
        for d in seed_dirs:
            mdir = d / "models" / name
            ckpt = mdir / "checkpoint.pt"
            cfg = mdir / "config.json"
            if not (ckpt.exists() and cfg.exists()):
                continue
            seed = d.name.split("seed")[-1]
            try:
                config = json.load(open(cfg))
                model = build_model(config, train_cov, return_std)
                state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["model_state_dict"]
                model.load_state_dict(state)
                res = run_one(model, loaders, raw, sigma_std)
                candidates.append((seed, *res))
            except Exception as exc:  # keep going; report at the end
                print(f"  [skip] {name} seed{seed}: {type(exc).__name__}: {exc}")

        if not candidates:
            print(f"  [none] {name}: no usable checkpoints")
            continue

        seed, idx, r_p, var, es, hits, hit_rate = candidates[0]
        out = OUTDIR / f"fig_fhs_var_{name}.pdf"
        plot_fhs_var(
            dates=idx, r=r_p, var=var, es=es, hits=hits, alpha=ALPHA,
            title=f"{model_label(name)}: one-step-ahead FHS VaR/ES",
            crisis_windows="default", save_path=out, show=False,
        )
        print(f"  {name:24s} seed{seed} hit={hit_rate:.2%}")
        summary.append((name, seed, hit_rate, out))

    print("\nsaved figures:")
    for name, seed, hr, out in summary:
        print(f"  {out.relative_to(ROOT)}  (seed{seed}, hit {hr:.2%})")


if __name__ == "__main__":
    main()
