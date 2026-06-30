"""Reconstruct one-step-ahead FHS VaR/ES figures for both BEKK benchmarks.

Self-contained illustration: reads the R-estimated conditional covariance
forecasts (train+val fitted and test) plus the return series, builds the
equal-weight portfolio, seeds the FHS history with the last `window`
standardized residuals from train+val, runs `fhs_var_es`, and plots the
resulting VaR/ES bands with violations.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from risk_metrics import fhs_var_es
from plots import plot_fhs_var

ASOF = "2026-05-03"
SEED = 42
TEST_LENGTH = 925
ALPHA = 0.01
WINDOW = 1000
ASSETS = ["^GSPC", "GC=F", "CL=F", "^TNX"]
N = len(ASSETS)
W = np.full(N, 1.0 / N)  # equal weights

R = ROOT / "r" / "output"
DATA = ROOT / "data"
HCOLS = [f"h_{i}{j}" for i in range(1, N + 1) for j in range(1, N + 1)]


def load_cov(path):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    H = df[HCOLS].to_numpy().reshape(-1, N, N)
    return df.index, H


def portfolio_series(returns_df, dates, H):
    """Align returns to forecast dates; return (r_p, vol_p)."""
    r = returns_df.reindex(dates)[ASSETS].to_numpy()
    r_p = r @ W
    vol_p = np.sqrt(np.einsum("i,tij,j->t", W, H, W))
    return r_p, vol_p


def plot_specification(specification, rets):
    tv_dates, H_tv = load_cov(
        R / f"bekk_forecasts_fitted_train_val_{specification}_asof_{ASOF}_seed{SEED}.csv"
    )
    te_dates, H_te = load_cov(
        R / f"bekk_forecasts_{specification}_asof_{ASOF}_seed{SEED}.csv"
    )
    te_dates, H_te = te_dates[-TEST_LENGTH:], H_te[-TEST_LENGTH:]

    # standardized portfolio residuals on train+val -> FHS seed history
    rp_tv, vol_tv = portfolio_series(rets, tv_dates, H_tv)
    z_tv = rp_tv / np.clip(vol_tv, 1e-12, None)
    z_tv = z_tv[np.isfinite(z_tv)]

    # test split
    rp_te, vol_te = portfolio_series(rets, te_dates, H_te)

    var, es, hits = fhs_var_es(
        z_hist_init=z_tv,
        r_oos=rp_te,
        vol_oos=vol_te,
        alpha=ALPHA,
        window=WINDOW,
    )

    out = ROOT / "figures" / "vares_plots" / f"fig_fhs_var_bekk_{specification}.pdf"
    hit_rate = plot_fhs_var(
        dates=te_dates,
        r=rp_te,
        var=var,
        es=es,
        hits=hits,
        alpha=ALPHA,
        title=(
            f"{specification.title()} BEKK: one-step-ahead FHS VaR/ES "
            "(equal-weight portfolio)"
        ),
        crisis_windows="default",  # canonical thesis windows (section3_eda)
        save_path=out,
        show=False,
    )
    print(f"hit rate = {hit_rate:.3%} (target {ALPHA:.1%}) over {len(rp_te)} test days")
    print(f"saved -> {out}")


def main():
    # returns: train+val+test live in the split csvs; concat the raw returns
    train = pd.read_csv(DATA / f"train_df_{ASOF}.csv", parse_dates=["Date"]).set_index("Date")
    val = pd.read_csv(DATA / f"val_df_{ASOF}.csv", parse_dates=["Date"]).set_index("Date")
    test = pd.read_csv(DATA / f"test_df_{ASOF}.csv", parse_dates=["Date"]).set_index("Date")
    rets = pd.concat([train, val, test])

    for specification in ("symmetric", "asymmetric"):
        plot_specification(specification, rets)


if __name__ == "__main__":
    main()
