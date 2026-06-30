"""Build the robustness-check appendix tables (LaTeX) from the evaluation CSVs.

Writes four compact tables into the Overleaf project's ``tables/`` directory:
  tab_robust_qlike.tex       portfolio vs matrix QLIKE (4-asset main, 11 seeds)
  tab_robust_parametric.tex  parametric VaR/ES backtests (alpha=1%, 11 seeds)
  tab_robust_alpha05.tex     FZ0/MCS ranking at alpha=5% (11 seeds)
  tab_robust_2asset.tex      FZ0/MCS on the S&P500/10Y cohorts (1962 vs 1990)

Run from the repo root:  python scripts/make_robustness_appendix_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # .../Masterthesis/Code/neural_bekk
EVAL = ROOT / "results"
OUT = ROOT.parent.parent / "overleaf" / "tables"        # .../Masterthesis/overleaf/tables


def to_tex(df: pd.DataFrame, path: Path) -> None:
    df.index.name = "Model"
    df.to_latex(path, float_format="%.4f", na_rep="--", escape=True)
    print(f"wrote {path}  ({df.shape[0]}x{df.shape[1]})")


def qlike_table() -> None:
    base = EVAL / "evaluation_qlike" / "2026-05-03"
    pf = pd.read_csv(base / "qlike_portfolio_summary.csv", index_col=0)
    mat = pd.read_csv(base / "qlike_matrix_summary.csv", index_col=0)
    out = pd.DataFrame({
        "Portfolio QLIKE": pf["mean_qlike"],
        "MCS pf": pf["in_mcs"],
        "Matrix QLIKE": mat["mean_qlike"].reindex(pf.index),
        "MCS mat": mat["in_mcs"].reindex(pf.index),
    }).sort_values("Portfolio QLIKE")
    to_tex(out, OUT / "tab_robust_qlike.tex")


def parametric_table() -> None:
    base = EVAL / "evaluation_parametric" / "asof_2026-05-03"
    t1 = pd.read_csv(base / "table1_main.csv").set_index("Model")
    t2 = pd.read_csv(base / "table2_var_backtests.csv").set_index("Model")
    out = pd.DataFrame({
        "FZ0 loss": t1["FZ0 loss"],
        "Hit rate": t1["Hit rate"],
        "Kupiec": t2["Kupiec"],
        "Chr. CC": t2["Chr. CC"],
        "DQ": t2["DQ"],
        "MCS (90\\%)": t1["MCS (90%)"],
    }).loc[t1.index]
    to_tex(out, OUT / "tab_robust_parametric.tex")


def alpha05_table() -> None:
    base = EVAL / "evaluation_alpha05" / "asof_2026-05-03"
    t1 = pd.read_csv(base / "table1_main.csv").set_index("Model")
    out = t1[["FZ0 loss", "Hit rate", "MCS (90%)"]].rename(
        columns={"MCS (90%)": "MCS (90\\%)"})
    to_tex(out, OUT / "tab_robust_alpha05.tex")


def two_asset_table() -> None:
    g = EVAL / "evaluation_gspc_tnx"
    a = pd.read_csv(g / "asof_gspc_tnx_1962" / "table1_main.csv").set_index("Model")
    b = pd.read_csv(g / "asof_gspc_tnx_1990" / "table1_main.csv").set_index("Model")
    out = pd.DataFrame({
        "FZ0 (1962)": a["FZ0 loss"],
        "MCS (1962)": a["MCS (90%)"],
        "FZ0 (1990)": b["FZ0 loss"].reindex(a.index),
        "MCS (1990)": b["MCS (90%)"].reindex(a.index),
    }).sort_values("FZ0 (1990)")
    to_tex(out, OUT / "tab_robust_2asset.tex")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qlike_table()
    parametric_table()
    alpha05_table()
    two_asset_table()


if __name__ == "__main__":
    main()
