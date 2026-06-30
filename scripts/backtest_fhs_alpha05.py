"""Recompute the per-seed FHS VaR/ES backtests at alpha = 5% (parallel output).

This re-runs the *evaluation* half of ``run_models_code_only.py`` without any
re-training: it reloads every seed's saved checkpoints + the R-estimated
covariance forecasts, rebuilds the equal-weight portfolio, runs filtered
historical simulation (FHS) at ``ALPHA`` and the full backtest suite
(``run_backtest_suite`` + DM plan + FZ-loss matrix), and writes a fresh
``all_backtest_results.json`` / ``fz_loss_matrix.csv`` per seed.

The methodology mirrors the monolith exactly:
  - neural test path = ``test_df[LOOKBACK:]`` (per-split SequenceDataset),
  - R models are aligned onto that same path (``H_test[LOOKBACK:]``),
  - FHS history = standardized train+val residuals, rolling ``WINDOW`` tail,
so the only thing that changes vs. the committed alpha=1% runs is ``ALPHA``.

The realized portfolio return path is identical across all 12 models by
construction (one canonical ``r_test``); only ``var``/``es``/``vol`` differ.

Outputs go to a parallel tree so the alpha=1% artifacts stay untouched:
    results/runs_alpha05/asof_<ASOF>_seed<k>/all_backtest_results.json
                                            /fz_loss_matrix.csv
                                            /fz_loss_summary.csv

Afterwards aggregate with:
    python evaluate_runs.py --runs-dir results/runs_alpha05 \
                            --out results/evaluation_alpha05 --no-overleaf

Requires the same R/rpy2 environment as the original pipeline (the CC/ER/ESR
expected-shortfall backtests call rugarch via rpy2).

Run from the repo root:  python scripts/backtest_fhs_alpha05.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from risk_metrics import fhs_var_es
from backtesting import (
    run_backtest_suite,
    run_fz_dm_comparison_plan,
    build_fz_loss_matrix,
)
from model_names import is_legacy_alias

# Reuse the validated offline reconstruction helpers (data, model build, Sigma_t).
from make_fhs_var_allmodels import (
    ASOF,
    ASSETS,
    LOOKBACK,
    WINDOW,
    build_model,
    load_splits,
    make_loader,
    predict_sigma_real,
)

# alpha-level for the FHS recompute. Default 0.05 (the original use); set
# FHS_ALPHA=0.01 to recompute at the thesis level (e.g. to fold a newly trained
# model such as neural_bekk_asym_full into the FZ0/MCS/DM tables).
ALPHA = float(os.environ.get("FHS_ALPHA", "0.05"))
BENCHMARK = "bekk_symmetric"

N = len(ASSETS)
W = np.full(N, 1.0 / N)  # equal-weight portfolio (float64)

DATA = ROOT / "data"
RUNS = ROOT / "results" / "runs"
ROUT = ROOT / "r" / "output"
OUT_RUNS = ROOT / "results" / f"runs_alpha{round(ALPHA * 100):02d}"  # 0.05->runs_alpha05, 0.01->runs_alpha01

HCOLS = [f"h_{i}{j}" for i in range(1, N + 1) for j in range(1, N + 1)]

# R model name -> (forecast file prefix, R kind tag)
R_MODELS = {
    "bekk_symmetric": ("bekk", "symmetric"),
    "bekk_asymmetric": ("bekk", "asymmetric"),
    "dcc": ("dcc", "dcc"),
    "adcc": ("dcc", "adcc"),
}


# --------------------------------------------------------------------------- #
# R covariance forecasts
# --------------------------------------------------------------------------- #
def _load_cov(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    H = df[HCOLS].to_numpy().reshape(-1, N, N)
    return df.index, H


def _portfolio_vol(dates, H, rets: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """Return (r_p over dates, vol_p Series indexed by dates)."""
    r = rets.reindex(dates)[ASSETS].to_numpy()
    r_p = r @ W
    vol_p = np.sqrt(np.einsum("i,tij,j->t", W, H, W))
    return r_p, pd.Series(vol_p, index=dates)


def r_model_vol_and_hist(prefix: str, kind: str, seed: str, rets: pd.DataFrame,
                         canonical_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (vol_test aligned to canonical_dates, FHS seed history z)."""
    tv = ROUT / f"{prefix}_forecasts_fitted_train_val_{kind}_asof_{ASOF}_seed{seed}.csv"
    te = ROUT / f"{prefix}_forecasts_{kind}_asof_{ASOF}_seed{seed}.csv"

    tv_dates, H_tv = _load_cov(tv)
    rp_tv, vol_tv = _portfolio_vol(tv_dates, H_tv, rets)
    z_hist = rp_tv / np.clip(vol_tv.to_numpy(), 1e-12, None)
    z_hist = z_hist[np.isfinite(z_hist)]

    te_dates, H_te = _load_cov(te)
    _, vol_te = _portfolio_vol(te_dates, H_te, rets)
    vol_test = vol_te.reindex(canonical_dates).to_numpy()
    return vol_test, z_hist


# --------------------------------------------------------------------------- #
# Neural covariance forecasts
# --------------------------------------------------------------------------- #
def _split_portfolio(real_df: pd.DataFrame, sigma_real: np.ndarray):
    idx = real_df.index[LOOKBACK:]
    r = real_df.loc[idx, ASSETS].to_numpy()
    r_p = r @ W
    vol_p = np.sqrt(np.clip(np.einsum("i,tij,j->t", W, sigma_real, W), 1e-24, None))
    return r_p, vol_p


def neural_vol_and_hist(model, loaders, raw, sigma_std) -> tuple[np.ndarray, np.ndarray]:
    """Return (vol_test, FHS seed history z = standardized train+val residuals)."""
    z_parts = []
    vol_test = None
    for split in ("train", "val", "test"):
        sigma_real = predict_sigma_real(model, loaders[split], sigma_std)
        r_p, vol_p = _split_portfolio(raw[split], sigma_real)
        if split in ("train", "val"):
            z_parts.append(r_p / np.clip(vol_p, 1e-12, None))
        else:
            vol_test = vol_p
    return vol_test, np.concatenate(z_parts)


# --------------------------------------------------------------------------- #
def main() -> None:
    raw, norm, sigma_std, train_cov, return_std = load_splits()
    loaders = {split: make_loader(norm[split]) for split in norm}

    # Canonical realized portfolio return path (seed-independent): test_df[LOOKBACK:].
    canonical_dates = raw["test"].index[LOOKBACK:]
    r_test = raw["test"].loc[canonical_dates, ASSETS].to_numpy() @ W

    # FHS history for the R models uses the raw train+val+test returns.
    rets = pd.concat([raw["train"], raw["val"], raw["test"]])

    seed_dirs = sorted(RUNS.glob(f"asof_{ASOF}_seed*"))
    if not seed_dirs:
        raise SystemExit(f"no run dirs under {RUNS}/asof_{ASOF}_seed*")

    for d in seed_dirs:
        seed = d.name.split("seed")[-1]
        print(f"\n=== seed {seed} ===")
        forecasts: dict[str, dict[str, np.ndarray]] = {}
        results: dict[str, dict] = {}

        # --- neural models ------------------------------------------------- #
        neural_names = sorted(
            p.name for p in (d / "models").glob("*")
            if p.is_dir() and not is_legacy_alias(p.name)
        )
        for name in neural_names:
            mdir = d / "models" / name
            ckpt, cfg = mdir / "checkpoint.pt", mdir / "config.json"
            if not (ckpt.exists() and cfg.exists()):
                continue
            try:
                import torch
                config = json.load(open(cfg))
                model = build_model(config, train_cov, return_std)
                state = torch.load(ckpt, map_location="cpu",
                                   weights_only=False)["model_state_dict"]
                model.load_state_dict(state)
                vol_test, z_hist = neural_vol_and_hist(model, loaders, raw, sigma_std)
            except Exception as exc:
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
                continue
            var, es, hits = fhs_var_es(z_hist_init=z_hist, r_oos=r_test,
                                       vol_oos=vol_test, alpha=ALPHA, window=WINDOW)
            forecasts[name] = {"returns": r_test, "var": var, "es": es}
            results[name] = run_backtest_suite(returns=r_test, var=var, es=es,
                                               alpha=ALPHA, volatility=vol_test)
            print(f"  {name:24s} hit={hits.mean():.2%}")

        # --- R models (BEKK / DCC) ----------------------------------------- #
        for name, (prefix, kind) in R_MODELS.items():
            try:
                vol_test, z_hist = r_model_vol_and_hist(prefix, kind, seed, rets,
                                                        canonical_dates)
            except FileNotFoundError as exc:
                print(f"  [skip] {name}: missing {Path(exc.filename).name}")
                continue
            var, es, hits = fhs_var_es(z_hist_init=z_hist, r_oos=r_test,
                                       vol_oos=vol_test, alpha=ALPHA, window=WINDOW)
            forecasts[name] = {"returns": r_test, "var": var, "es": es}
            results[name] = run_backtest_suite(returns=r_test, var=var, es=es,
                                               alpha=ALPHA, volatility=vol_test)
            print(f"  {name:24s} hit={hits.mean():.2%}")

        if BENCHMARK not in forecasts:
            print(f"  [warn] benchmark {BENCHMARK!r} missing; skipping DM/MCS")
            continue

        # --- forecast comparison + FZ-loss matrix -------------------------- #
        dm = run_fz_dm_comparison_plan(model_forecasts=forecasts, alpha=ALPHA,
                                       benchmark=BENCHMARK, structured_pairs=(),
                                       include_pairwise=False)
        names, loss_matrix = build_fz_loss_matrix(model_forecasts=forecasts, alpha=ALPHA)
        fz = pd.DataFrame(loss_matrix, columns=names)

        out_dir = OUT_RUNS / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        fz_path = out_dir / "fz_loss_matrix.csv"
        fz.to_csv(fz_path, index=False)
        summary = (fz.mean().rename("mean_fz_loss").sort_values()
                   .reset_index().rename(columns={"index": "model"}))
        summary["rank"] = np.arange(1, len(summary) + 1)
        summary_path = out_dir / "fz_loss_summary.csv"
        summary.to_csv(summary_path, index=False)

        all_results = {
            **results,
            "forecast_comparison": dm,
            "mcs_input": {
                "loss_function": "fissler_ziegel_patton_ziegel_chen",
                "loss_matrix_csv": str(fz_path),
                "loss_summary_csv": str(summary_path),
                "n_obs": int(len(fz)),
                "models": list(names),
                "mean_fz_loss": {row["model"]: float(row["mean_fz_loss"])
                                 for _, row in summary.iterrows()},
            },
        }
        with open(out_dir / "all_backtest_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  -> {out_dir.relative_to(ROOT)}  ({len(results)} models)")


if __name__ == "__main__":
    main()
