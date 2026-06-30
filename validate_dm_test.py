"""Cross-validation of the in-house modified Diebold--Mariano test.

At the one-step-ahead horizon h=1 the Harvey--Leybourne--Newbold (1997)
modified DM statistic on a loss differential d_t is algebraically identical
to a one-sample t-test of H0: E[d_t]=0 (the long-run variance reduces to the
sample variance and the statistic is referred to a t_{n-1} distribution).
This script verifies that ``run_modified_dm_test`` reproduces that reference
on the actual FZ0 loss differentials, across all available seeds and models,
benchmarked against the symmetric BEKK.

Run:  python validate_dm_test.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from backtesting import run_modified_dm_test

BENCHMARK = "bekk_symmetric"
HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(HERE, "results", "runs", "asof_*_seed*", "fz_loss_matrix.csv")


def main() -> int:
    files = sorted(glob.glob(PATTERN))
    if not files:
        print(f"No loss matrices found under {PATTERN}")
        return 1

    max_stat_diff = 0.0
    max_p_diff = 0.0
    n_comparisons = 0

    for path in files:
        seed = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        if BENCHMARK not in df.columns:
            print(f"  [skip] {seed}: no '{BENCHMARK}' column")
            continue
        bench = df[BENCHMARK].to_numpy(dtype=float)
        for model in df.columns:
            if model == BENCHMARK:
                continue
            d = df[model].to_numpy(dtype=float) - bench
            d = d[np.isfinite(d)]
            if np.allclose(d, 0.0) or np.var(d) == 0.0:
                continue  # deterministic-vs-deterministic, no variance

            own = run_modified_dm_test(d, h=1)
            ref = ttest_1samp(d, popmean=0.0)  # two-sided, df = n-1

            stat_diff = abs(own["modified_dm_stat"] - float(ref.statistic))
            p_diff = abs(own["p_value"] - float(ref.pvalue))
            max_stat_diff = max(max_stat_diff, stat_diff)
            max_p_diff = max(max_p_diff, p_diff)
            n_comparisons += 1

    print(f"Compared {n_comparisons} model/seed loss differentials")
    print(f"  max |stat_own - stat_ttest| = {max_stat_diff:.3e}")
    print(f"  max |p_own    - p_ttest|    = {max_p_diff:.3e}")

    tol = 1e-10
    ok = max_stat_diff < tol and max_p_diff < tol
    print("RESULT:", "PASS - implementations agree to numerical precision"
          if ok else "FAIL - discrepancy exceeds tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
