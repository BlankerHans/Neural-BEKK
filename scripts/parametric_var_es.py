"""Robustness check: parametric VaR/ES instead of filtered historical simulation.

For every model the conditional portfolio variance h_{p,t}=w'Sigma_t w is mapped
to VaR/ES through a *law-matched* parametric quantile rather than the empirical
FHS quantile:

  - Gaussian models (BEKK, DCC, ADCC, LSTM/GRU Normal):
        VaR = sigma * Phi^{-1}(alpha),   ES = sigma * (-phi(Phi^{-1}(alpha))/alpha)
  - Student-t models (LSTM/GRU Student-t, BEKK-LSTM, Neural-BEKK), with the
    per-seed estimated nu from the model config:
        scale = sigma * sqrt((nu-2)/nu)            # Sigma_t is the COVARIANCE
        VaR   = scale * t_nu^{-1}(alpha)
        ES    = scale * ( -(nu + t_q^2)/(nu-1) * f_nu(t_q)/alpha ),  t_q=t_nu^{-1}(alpha)

Sigma_t is reconstructed exactly as in backtest_fhs_alpha05.py (neural
checkpoints + R outputs); only the VaR/ES map changes. The same backtest suite,
FZ0-loss matrix and DM plan are then produced, and aggregated with evaluate_runs.

    python scripts/parametric_var_es.py
    python evaluate_runs.py --runs-dir results/runs_parametric \
                            --out results/evaluation_parametric --no-overleaf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, t as t_dist

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from model_names import is_legacy_alias
from backtesting import run_backtest_suite, run_fz_dm_comparison_plan, build_fz_loss_matrix
from make_fhs_var_allmodels import ASOF, ASSETS, LOOKBACK, build_model, load_splits, make_loader
from backtest_fhs_alpha05 import neural_vol_and_hist, r_model_vol_and_hist, R_MODELS

ALPHA = 0.01
BENCHMARK = "bekk_symmetric"
N = len(ASSETS)
W = np.full(N, 1.0 / N)

RUNS = ROOT / "results" / "runs"
OUT_RUNS = ROOT / "results" / "runs_parametric"


def parametric_var_es(sigma: np.ndarray, alpha: float, dist: str, nu: float | None):
    """Return (var, es) on the return scale (both negative); sigma is the
    conditional portfolio volatility from the covariance forecast."""
    sigma = np.asarray(sigma, dtype=float)
    if dist == "gaussian":
        z = norm.ppf(alpha)
        var = sigma * z
        es = sigma * (-norm.pdf(z) / alpha)
    elif dist == "student_t":
        if nu is None or nu <= 2:
            raise ValueError(f"student_t needs nu>2, got {nu}")
        scale = sigma * np.sqrt((nu - 2.0) / nu)        # Sigma_t is covariance -> scale
        tq = t_dist.ppf(alpha, df=nu)
        var = scale * tq
        es = scale * (-((nu + tq ** 2) / (nu - 1.0)) * t_dist.pdf(tq, df=nu) / alpha)
    else:
        raise ValueError(f"unknown dist {dist!r}")
    return var, es


def main() -> None:
    raw, norm_df, sigma_std, train_cov, return_std = load_splits()
    loaders = {s: make_loader(norm_df[s]) for s in norm_df}
    canonical = raw["test"].index[LOOKBACK:]
    r_test = raw["test"].loc[canonical, ASSETS].to_numpy(dtype=float) @ W

    seed_dirs = sorted(RUNS.glob(f"asof_{ASOF}_seed*"))
    if not seed_dirs:
        raise SystemExit(f"no run dirs under {RUNS}/asof_{ASOF}_seed*")

    for d in seed_dirs:
        seed = d.name.split("seed")[-1]
        print(f"\n=== seed {seed} ===")
        forecasts: dict[str, dict] = {}
        results: dict[str, dict] = {}

        # neural models: vol from checkpoint, law/nu from config
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
                vol, _ = neural_vol_and_hist(model, loaders, raw, sigma_std)
                dist = config.get("distribution", "gaussian")
                nu = config.get("nu")
            except Exception as exc:
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
                continue
            var, es = parametric_var_es(vol, ALPHA, dist, nu)
            forecasts[name] = {"returns": r_test, "var": var, "es": es}
            results[name] = run_backtest_suite(returns=r_test, var=var, es=es,
                                               alpha=ALPHA, volatility=vol)
            print(f"  {name:24s} {dist:9s} nu={nu if nu else '-':<6} hit={np.mean(r_test<=var):.2%}")

        # R econometric models: Gaussian QML, vol from R covariance forecasts
        for name, (prefix, kind) in R_MODELS.items():
            try:
                vol, _ = r_model_vol_and_hist(prefix, kind, seed, pd.concat(
                    [raw["train"], raw["val"], raw["test"]]), canonical)
            except FileNotFoundError:
                continue
            var, es = parametric_var_es(vol, ALPHA, "gaussian", None)
            forecasts[name] = {"returns": r_test, "var": var, "es": es}
            results[name] = run_backtest_suite(returns=r_test, var=var, es=es,
                                               alpha=ALPHA, volatility=vol)
            print(f"  {name:24s} gaussian  nu=-      hit={np.mean(r_test<=var):.2%}")

        if BENCHMARK not in forecasts:
            print(f"  [warn] benchmark missing; skipping DM/MCS for seed {seed}")
            continue

        dm = run_fz_dm_comparison_plan(model_forecasts=forecasts, alpha=ALPHA,
                                       benchmark=BENCHMARK, structured_pairs=(),
                                       include_pairwise=False)
        names, loss_matrix = build_fz_loss_matrix(model_forecasts=forecasts, alpha=ALPHA)
        fz = pd.DataFrame(loss_matrix, columns=names)

        od = OUT_RUNS / d.name
        od.mkdir(parents=True, exist_ok=True)
        fz_path = od / "fz_loss_matrix.csv"
        fz.to_csv(fz_path, index=False)
        summary = (fz.mean().rename("mean_fz_loss").sort_values()
                   .reset_index().rename(columns={"index": "model"}))
        summary["rank"] = np.arange(1, len(summary) + 1)
        summary.to_csv(od / "fz_loss_summary.csv", index=False)

        all_results = {
            **results,
            "forecast_comparison": dm,
            "mcs_input": {
                "loss_function": "fissler_ziegel_patton_ziegel_chen",
                "loss_matrix_csv": str(fz_path),
                "n_obs": int(len(fz)),
                "models": list(names),
                "mean_fz_loss": {row["model"]: float(row["mean_fz_loss"])
                                 for _, row in summary.iterrows()},
            },
        }
        with open(od / "all_backtest_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  -> {od.relative_to(ROOT)}  ({len(results)} models)")


if __name__ == "__main__":
    main()
