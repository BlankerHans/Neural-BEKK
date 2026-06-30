"""Aggregate backtest results across seeds and generate thesis-ready tables.

Reads all ``results/runs/<asof>_seed<k>/all_backtest_results.json`` (plus the
per-run ``fz_loss_matrix.csv``), aggregates across seeds, runs the Model
Confidence Set per seed, and writes per-as-of outputs to
``results/evaluation/<asof>/``:

- ``tidy_metrics.csv``        long format (seed, model, metric, value)
- ``table1_main.csv|.tex``    headline table: FZ loss, hit rate, MCS
- ``table2_var_backtests.csv|.tex``
- ``table3_es_backtests.csv|.tex``
- ``table4_dm_benchmark.csv|.tex``
- ``fig_fz_loss_seeds.pdf``   FZ loss distribution across seeds per model
- ``fig_cum_loss_diff.pdf``   cumulative FZ loss differential vs benchmark
- ``per_seed_pvalues.csv``    appendix: every p-value for every seed
- ``report.md``               human-readable summary

For the most recent as-of date, the ``.tex`` tables and ``.pdf`` figures are
additionally copied into the Overleaf project (``--overleaf-dir``) so the
thesis picks them up on the next compile.

Usage:
    python evaluate_runs.py [--runs-dir results/runs] [--out results/evaluation]
                            [--overleaf-dir ../overleaf | --no-overleaf]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

# Reuse the thesis-wide matplotlib style defined for the EDA figures so the
# evaluation plots match the rest of the thesis (serif fonts, muted grid, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from section3_eda import set_thesis_style

# Publication-ready model labels are the single source of truth in model_names.py.
from model_names import MODEL_LABELS

TEST_LEVEL = 0.05
MCS_SIZE = 0.10  # 90% model confidence set
MCS_REPS = 1000
MCS_SEED = 42
BENCHMARK = "bekk_symmetric"
DEFAULT_OVERLEAF = Path(__file__).resolve().parent.parent.parent / "overleaf"

NON_MODEL_KEYS = {"forecast_comparison", "neural_bekk_artifacts", "mcs_input"}

COLUMN_LABELS = {
    "fz_loss_mean": "FZ0 loss",
    "fz_loss_std": "Std (seeds)",
    "hit_rate_mean": "Hit rate",
    "mcs90_inclusion": "MCS (90%)",
    "mcs_pval_mean": "MCS p-value",
    "n_exceptions_mean": "Exceptions",
    "expected_exceptions": "Expected",
    "rej_kupiec": "Kupiec",
    "rej_christoffersen_ind": "Chr. ind.",
    "rej_christoffersen_cc": "Chr. CC",
    "rej_dq": "DQ",
    "rej_er": "ER",
    "rej_cc": "CC",
    "rej_esr_v1": "ESR (v1)",
    "rej_esr_v2": "ESR (v2)",
    "rej_esr_v3": "ESR (v3)",
    "mean_loss_differential": "Mean loss diff.",
    "seeds_model_preferred": "Preferred (seeds)",
    "seeds_dm_significant": "p<0.05 (seeds)",
    "pval_dm_mean": "Mean p-value",
}


def discover_runs(runs_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group run directories by as-of date, sorted by seed."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in runs_dir.iterdir():
        match = re.fullmatch(r"(asof_.+)_seed(\d+)", path.name)
        if match and (path / "all_backtest_results.json").exists():
            groups[match.group(1)].append((int(match.group(2)), path))
    return {asof: sorted(runs) for asof, runs in groups.items()}


def extract_model_metrics(model: str, res: dict[str, Any]) -> dict[str, float]:
    """Flatten one model's backtest results into scalar metrics."""
    var = res["var"]
    return {
        "hit_rate": var["kupiec"]["hit_rate"],
        "n_exceptions": var["kupiec"]["n_exceptions"],
        "expected_exceptions": var["kupiec"]["expected_exceptions"],
        "pval_kupiec": var["kupiec"]["p_value"],
        "pval_christoffersen_ind": var["christoffersen"]["p_value_ind"],
        "pval_christoffersen_cc": var["christoffersen"]["p_value_cc"],
        "pval_dq": var["dq"]["p_value"],
        "pval_er": res["er"]["pvalue_onesided_standardized"],
        "pval_cc": res["cc"]["pvalue_onesided_general"],
        "pval_esr_v1": res["esr"]["v1"]["pvalue_twosided_asymptotic"],
        "pval_esr_v2": res["esr"]["v2"]["pvalue_twosided_asymptotic"],
        "pval_esr_v3": res["esr"]["v3"]["pvalue_onesided_asymptotic"],
    }


def load_tidy(runs: list[tuple[int, Path]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Return (per-model metrics, DM-vs-benchmark results, loss matrices)."""
    metric_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    loss_matrices: dict[int, pd.DataFrame] = {}

    for seed, run_dir in runs:
        with open(run_dir / "all_backtest_results.json") as f:
            data = json.load(f)

        for model, res in data.items():
            if model in NON_MODEL_KEYS:
                continue
            row = {"seed": seed, "model": model, "n_obs": res["meta"]["n_obs"],
                   "alpha": res["meta"]["alpha"]}
            row.update(extract_model_metrics(model, res))
            row["mean_fz_loss"] = data["mcs_input"]["mean_fz_loss"][model]
            metric_rows.append(row)

        for test in data["forecast_comparison"]["benchmark"].values():
            dm_rows.append({
                "seed": seed,
                "model": test["model_1"],
                "benchmark": test["model_2"],
                "mean_loss_differential": test["mean_loss_differential"],
                "pval_dm": test["p_value"],
                "preferred": test["preferred_by_mean_loss"],
            })

        loss_csv = run_dir / "fz_loss_matrix.csv"
        if loss_csv.exists():
            loss_matrices[seed] = pd.read_csv(loss_csv)

    return pd.DataFrame(metric_rows), pd.DataFrame(dm_rows), loss_matrices


def run_mcs_per_seed(loss_matrices: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Run the Model Confidence Set per seed; return inclusion flags and p-values."""
    rows: list[dict[str, Any]] = []
    for seed, losses in sorted(loss_matrices.items()):
        mcs = MCS(losses, size=MCS_SIZE, reps=MCS_REPS, bootstrap="stationary",
                  seed=MCS_SEED)
        mcs.compute()
        included = set(mcs.included)
        pvalues = mcs.pvalues["Pvalue"]
        for model in losses.columns:
            rows.append({
                "seed": seed,
                "model": model,
                "in_mcs": model in included,
                "pval_mcs": float(pvalues.get(model, np.nan)),
            })
    return pd.DataFrame(rows)


def deterministic_models(metrics: pd.DataFrame) -> set[str]:
    """Models whose mean FZ loss is identical across all seeds (no training noise)."""
    spread = metrics.groupby("model")["mean_fz_loss"].agg(lambda s: s.max() - s.min())
    return set(spread[spread == 0].index)


def rejection_rates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Share of seeds in which each test rejects at TEST_LEVEL, per model."""
    pval_cols = [c for c in metrics.columns if c.startswith("pval_")]
    rej = metrics.groupby("model")[pval_cols].agg(lambda s: (s < TEST_LEVEL).mean())
    return rej.rename(columns=lambda c: c.replace("pval_", "rej_"))


def fmt_count(rate: float, n: int) -> str:
    return f"{round(rate * n)}/{n}"


def prettify(table: pd.DataFrame) -> pd.DataFrame:
    """Apply publication-ready model and column names for export."""
    out = table.rename(index=MODEL_LABELS, columns=COLUMN_LABELS)
    out.index.name = "Model"
    return out


def to_markdown_table(table: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (avoids tabulate dependency)."""
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "--" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    header = [table.index.name or "model", *table.columns]
    rows = [[fmt(idx), *(fmt(v) for v in row)] for idx, row in table.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    sep = ["-" * w for w in widths]
    lines = [header, sep, *rows]
    return "\n".join("| " + " | ".join(c.ljust(w) for c, w in zip(line, widths)) + " |"
                     for line in lines)


def build_tables(metrics: pd.DataFrame, dm: pd.DataFrame, mcs: pd.DataFrame,
                 ) -> dict[str, pd.DataFrame]:
    n_seeds = metrics["seed"].nunique()
    det = deterministic_models(metrics)
    rej = rejection_rates(metrics)

    grouped = metrics.groupby("model")
    fz = grouped["mean_fz_loss"].agg(["mean", "std", "min", "max"])
    fz.loc[list(det & set(fz.index)), "std"] = np.nan
    hit = grouped["hit_rate"].mean()

    mcs_grouped = mcs.groupby("model")
    mcs_incl = mcs_grouped["in_mcs"].mean()
    mcs_pval = mcs_grouped["pval_mcs"].mean()

    order = fz["mean"].sort_values().index

    table1 = pd.DataFrame({
        "fz_loss_mean": fz["mean"],
        "fz_loss_std": fz["std"],
        "hit_rate_mean": hit,
        "mcs90_inclusion": mcs_incl.map(lambda r: fmt_count(r, n_seeds)),
        "mcs_pval_mean": mcs_pval,
    }).loc[order]

    table2 = pd.DataFrame({
        "hit_rate_mean": hit,
        "n_exceptions_mean": grouped["n_exceptions"].mean(),
        "expected_exceptions": grouped["expected_exceptions"].first(),
        "rej_kupiec": rej["rej_kupiec"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_christoffersen_ind": rej["rej_christoffersen_ind"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_christoffersen_cc": rej["rej_christoffersen_cc"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_dq": rej["rej_dq"].map(lambda r: fmt_count(r, n_seeds)),
    }).loc[order]

    table3 = pd.DataFrame({
        "rej_er": rej["rej_er"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_cc": rej["rej_cc"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_esr_v1": rej["rej_esr_v1"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_esr_v2": rej["rej_esr_v2"].map(lambda r: fmt_count(r, n_seeds)),
        "rej_esr_v3": rej["rej_esr_v3"].map(lambda r: fmt_count(r, n_seeds)),
    }).loc[order]

    dm_grouped = dm.groupby("model")
    table4 = pd.DataFrame({
        "mean_loss_differential": dm_grouped["mean_loss_differential"].mean(),
        "seeds_model_preferred": dm_grouped.apply(
            lambda g: fmt_count((g["preferred"] == g.name).mean(), len(g)),
            include_groups=False),
        "seeds_dm_significant": dm_grouped["pval_dm"].agg(
            lambda s: fmt_count((s < TEST_LEVEL).mean(), len(s))),
        "pval_dm_mean": dm_grouped["pval_dm"].mean(),
    }).sort_values("mean_loss_differential")

    return {"table1_main": table1, "table2_var_backtests": table2,
            "table3_es_backtests": table3, "table4_dm_benchmark": table4}


def make_figures(metrics: pd.DataFrame, loss_matrices: dict[int, pd.DataFrame],
                 out_dirs: list[Path]) -> None:
    """Write fig_fz_loss_seeds.pdf and fig_cum_loss_diff.pdf to each out_dir."""
    set_thesis_style()
    det = deterministic_models(metrics)
    order = (metrics.groupby("model")["mean_fz_loss"].mean()
             .sort_values().index.tolist())

    # Figure 1: distribution of mean FZ loss across seeds per model.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rng = np.random.default_rng(0)
    for pos, model in enumerate(order):
        vals = metrics.loc[metrics["model"] == model, "mean_fz_loss"].to_numpy()
        if model in det:
            ax.plot(pos, vals[0], marker="D", color="0.3", markersize=5, zorder=3)
        else:
            ax.boxplot(vals, positions=[pos], widths=0.5, showfliers=False,
                       medianprops={"color": "C0"})
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.plot(pos + jitter, vals, linestyle="", marker="o", markersize=3,
                    color="C0", alpha=0.5, zorder=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in order],
                       rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean FZ0 loss (out-of-sample)")
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    handles = [plt.Line2D([], [], marker="D", linestyle="", color="0.3",
                          markersize=5, label="deterministic (single fit)"),
               plt.Line2D([], [], marker="o", linestyle="", color="C0",
                          markersize=4, label="per-seed result")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig_fz_loss_seeds.pdf")
    plt.close(fig)

    # Figure 2: cumulative FZ loss differential vs the benchmark, seed-averaged.
    diffs: dict[str, np.ndarray] = {}
    for model in order:
        if model == BENCHMARK:
            continue
        per_seed = [lm[model].to_numpy() - lm[BENCHMARK].to_numpy()
                    for lm in loss_matrices.values() if model in lm.columns]
        diffs[model] = np.cumsum(np.mean(per_seed, axis=0))

    highlight = [m for m in order if m != BENCHMARK][:3]
    highlight_colors = dict(zip(highlight, ["tab:blue", "tab:orange", "tab:green"]))
    # Pale but mutually distinct colors for the non-highlighted models; hues
    # chosen to stay clearly apart from the three highlight colors.
    pale_palette = ["#9e9ac8", "#fc9272", "#7fcdbb", "#dfc27d", "#c994c7",
                    "#f4a3c2", "#bdb76b", "#a5b8d0"]
    others = [m for m in diffs if m not in highlight]
    pale_colors = {m: pale_palette[i % len(pale_palette)]
                   for i, m in enumerate(others)}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Spread the right-edge labels of the non-highlighted lines so they don't
    # overlap (e.g. DCC and ADCC end at nearly identical values).
    endpoints = sorted((s[-1], m) for m, s in diffs.items() if m not in highlight)
    span = (min(s.min() for s in diffs.values()),
            max(s.max() for s in diffs.values()))
    min_gap = (span[1] - span[0]) / 24
    label_y: dict[str, float] = {}
    prev = -np.inf
    for y, model in endpoints:
        y = max(y, prev + min_gap)
        label_y[model] = y
        prev = y
    for model, series in diffs.items():
        if model in highlight:
            ax.plot(series, linewidth=1.8, label=MODEL_LABELS.get(model, model),
                    color=highlight_colors[model], zorder=3)
        else:
            c = pale_colors[model]
            label_c = tuple(0.7 * v for v in matplotlib.colors.to_rgb(c))
            ax.plot(series, linewidth=0.9, color=c, zorder=1)
            ax.annotate(MODEL_LABELS.get(model, model),
                        xy=(len(series) - 1, label_y[model]), fontsize=6,
                        color=label_c, xytext=(3, 0), textcoords="offset points",
                        va="center")
    ax.axhline(0.0, color="0.2", linewidth=0.8, linestyle="--")
    ax.set_ylim(span[0] - min_gap, max(span[1], prev) + min_gap)
    ax.set_xlabel("Out-of-sample observation")
    ax.set_ylabel(f"Cumulative FZ0 loss diff. vs {MODEL_LABELS[BENCHMARK]}")
    ax.set_xlim(0, len(next(iter(diffs.values()))) * 1.12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    ax.legend(fontsize=8, loc="lower left", title="seed average", title_fontsize=8)
    fig.tight_layout()
    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig_cum_loss_diff.pdf")
    plt.close(fig)


def write_report(out_dir: Path, asof: str, metrics: pd.DataFrame,
                 tables: dict[str, pd.DataFrame]) -> None:
    n_seeds = metrics["seed"].nunique()
    n_obs = int(metrics["n_obs"].iloc[0])
    alpha = metrics["alpha"].iloc[0]
    det = sorted(deterministic_models(metrics))
    table1 = tables["table1_main"]
    best = table1.index[0]

    lines = [
        f"# Evaluation report — {asof}",
        "",
        f"- Seeds: {n_seeds} ({sorted(int(s) for s in metrics['seed'].unique())})",
        f"- Out-of-sample observations: {n_obs}, VaR level alpha = {alpha}",
        f"- Backtest rejection level: {TEST_LEVEL}, MCS size: {MCS_SIZE} "
        f"({int((1 - MCS_SIZE) * 100)}% set, {MCS_REPS} bootstrap reps)",
        f"- Deterministic models (identical across seeds): {', '.join(det)}",
        "",
        f"Best mean FZ0 loss: **{best}** ({table1.loc[best, 'fz_loss_mean']:.4f}).",
        "",
        "Note: seeds share the same test window; the seed std measures training "
        "instability, not sampling uncertainty.",
        "",
    ]
    for name, table in tables.items():
        lines += [f"## {name}", "", to_markdown_table(prettify(table)), ""]
    (out_dir / "report.md").write_text("\n".join(lines))


def evaluate_asof(asof: str, runs: list[tuple[int, Path]], out_root: Path,
                  overleaf_dir: Path | None) -> Path:
    out_dir = out_root / asof
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, dm, loss_matrices = load_tidy(runs)
    mcs = run_mcs_per_seed(loss_matrices)
    tables = build_tables(metrics, dm, mcs)

    tidy = metrics.melt(id_vars=["seed", "model"], var_name="metric", value_name="value")
    tidy.to_csv(out_dir / "tidy_metrics.csv", index=False)

    pval_cols = ["seed", "model"] + [c for c in metrics.columns if c.startswith("pval_")]
    per_seed = metrics[pval_cols].merge(mcs[["seed", "model", "pval_mcs", "in_mcs"]],
                                        on=["seed", "model"], how="left")
    per_seed.sort_values(["model", "seed"]).to_csv(out_dir / "per_seed_pvalues.csv",
                                                   index=False)

    tex_dirs = [out_dir]
    fig_dirs = [out_dir]
    if overleaf_dir is not None:
        tex_dirs.append(overleaf_dir / "tables")
        fig_dirs.append(overleaf_dir / "images")
        for d in (overleaf_dir / "tables", overleaf_dir / "images"):
            d.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        pretty = prettify(table)
        pretty.to_csv(out_dir / f"{name}.csv")
        for tex_dir in tex_dirs:
            pretty.to_latex(tex_dir / f"{name}.tex", float_format="%.4f",
                            na_rep="--", escape=True)

    make_figures(metrics, loss_matrices, fig_dirs)
    write_report(out_dir, asof, metrics, tables)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--out", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--overleaf-dir", type=Path, default=DEFAULT_OVERLEAF,
                        help="Overleaf project root; tex tables go to tables/, "
                             "figures to images/ (latest as-of only)")
    parser.add_argument("--no-overleaf", action="store_true",
                        help="skip exporting tables/figures to the Overleaf project")
    args = parser.parse_args()

    groups = discover_runs(args.runs_dir)
    if not groups:
        raise SystemExit(f"No run directories found in {args.runs_dir}")
    latest = sorted(groups)[-1]
    for asof, runs in sorted(groups.items()):
        overleaf = None
        if asof == latest and not args.no_overleaf and args.overleaf_dir.exists():
            overleaf = args.overleaf_dir
        out_dir = evaluate_asof(asof, runs, args.out, overleaf)
        exported = f", exported to {overleaf}" if overleaf else ""
        print(f"{asof}: {len(runs)} seeds -> {out_dir}{exported}")


if __name__ == "__main__":
    main()
