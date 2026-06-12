"""
EDA plots and descriptive statistics for Section 3 of the thesis.

Functions:
    set_thesis_style           - consistent thesis/paper matplotlib styling
    plot_log_returns           - multi-panel time series of log returns
    descriptive_stats_table    - descriptive statistics table
    plot_squared_return_acf    - ACF of squared returns
    plot_rolling_correlations  - rolling pairwise correlations
    plot_qq_normal             - QQ-plots vs. normal distribution
    stats_to_latex             - LaTeX-ready descriptive table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


ASSET_LABELS = {
    "^GSPC": "S&P 500",
    "GC=F": "Gold",
    "CL=F": "Oil",
    "^TNX": "10Y Yield",
    "BTC-USD": "Bitcoin",
}

DEFAULT_CRISIS_WINDOWS = [
    ("Dot-com", "2000-03-10", "2002-10-09"),
    ("GFC", "2007-12-01", "2009-06-30"),
    ("Euro debt", "2010-05-01", "2012-07-31"),
    ("COVID", "2020-02-01", "2020-04-30"),
    ("2022 shock", "2022-01-01", "2022-10-31"),
    ("Iran conflict", "2026-02-28", None),
]


def _asset_label(name: str) -> str:
    return ASSET_LABELS.get(str(name), str(name))


def _shade_crisis_windows(
    ax,
    crisis_windows: list[tuple[str, str, str | None]] | None,
    annotate: bool = False,
    data_end: pd.Timestamp | None = None,
) -> None:
    if not crisis_windows:
        return

    for label, start, end in crisis_windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end is not None else data_end
        if end_ts is None:
            continue
        ax.axvspan(start_ts, end_ts, color="0.70", alpha=0.16, lw=0, zorder=0)
        if annotate:
            mid_ts = start_ts + (end_ts - start_ts) / 2
            ax.text(
                mid_ts,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="0.35",
            )


def set_thesis_style() -> None:
    """Set a restrained, publication-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.45,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig, save_path: str | Path | None) -> None:
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def _scaled_returns(logret: pd.DataFrame, scale: float) -> pd.DataFrame:
    return logret.astype(float) * float(scale)


def plot_log_returns(
    logret: pd.DataFrame,
    scale: float = 100.0,
    title: str = "Log returns",
    show_title: bool = False,
    crisis_windows: list[tuple[str, str, str | None]] | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
):
    """Multi-panel time series. One panel per asset, shared x-axis."""
    data = _scaled_returns(logret, scale)
    n = data.shape[1]

    if figsize is None:
        figsize = (12, 1.6 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    data_end = pd.Timestamp(data.index.max())

    for ax, col in zip(axes, data.columns):
        _shade_crisis_windows(ax, crisis_windows, annotate=ax is axes[0], data_end=data_end)
        ax.plot(data.index, data[col].values, color="black", lw=0.6)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_ylabel(_asset_label(col), rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")

    if show_title:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def descriptive_stats_table(
    logret: pd.DataFrame,
    scale: float = 100.0,
    lb_lags: int = 10,
) -> pd.DataFrame:
    """
    Descriptive statistics per asset.

    Mean, SD, Min, and Max are reported after applying ``scale``. Test
    statistics are computed on the original return series. Kurtosis is the
    Pearson definition, so the normal-distribution reference value is 3.
    """
    rows = {}

    for col in logret.columns:
        r_raw = logret[col].dropna().astype(float).values
        r = r_raw * scale
        r2_raw = r_raw**2

        jb_stat, jb_p = stats.jarque_bera(r_raw)
        adf_stat, adf_p, *_ = adfuller(r_raw, autolag="AIC")

        lb_r = acorr_ljungbox(r_raw, lags=[lb_lags], return_df=True).iloc[0]
        lb_r2 = acorr_ljungbox(r2_raw, lags=[lb_lags], return_df=True).iloc[0]

        rows[col] = {
            "N": len(r_raw),
            "Mean": np.mean(r),
            "SD": np.std(r, ddof=1),
            "Skew": stats.skew(r_raw, bias=False),
            "Kurtosis": stats.kurtosis(r_raw, fisher=False, bias=False),
            "Min": np.min(r),
            "Max": np.max(r),
            "JB": jb_stat,
            "JB p": jb_p,
            f"LB$_{{{lb_lags}}}(r_t)$": lb_r["lb_stat"],
            f"LB$_{{{lb_lags}}}(r_t)$ p": lb_r["lb_pvalue"],
            f"LB$_{{{lb_lags}}}(r_t^2)$": lb_r2["lb_stat"],
            f"LB$_{{{lb_lags}}}(r_t^2)$ p": lb_r2["lb_pvalue"],
            "ADF": adf_stat,
            "ADF p": adf_p,
        }

    return pd.DataFrame(rows).T


def plot_squared_return_acf(
    logret: pd.DataFrame,
    lags: int = 40,
    ncols: int = 2,
    title: str = "ACF of squared returns",
    show_title: bool = False,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
):
    """ACF of squared returns per asset."""
    n = logret.shape[1]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (5.5 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, logret.columns):
        r2 = logret[col].dropna().astype(float).values**2
        plot_acf(r2, lags=lags, ax=ax, title=f"{_asset_label(col)}: ACF($r_t^2$)", zero=False)
        ax.grid(alpha=0.25)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if show_title:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_rolling_correlations(
    logret: pd.DataFrame,
    window: int = 120,
    pairs: list[tuple[str, str]] | None = None,
    title: str | None = None,
    show_title: bool = False,
    crisis_windows: list[tuple[str, str, str | None]] | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
):
    """Rolling correlations for selected or all asset pairs."""
    cols = list(logret.columns)

    if pairs is None:
        pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("At least one pair is required for rolling correlations.")

    if figsize is None:
        figsize = (12, 1.7 * n_pairs)

    fig, axes = plt.subplots(n_pairs, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    data_end = pd.Timestamp(logret.index.max())

    for ax, (a, b) in zip(axes, pairs):
        rho_t = logret[a].rolling(window).corr(logret[b])
        rho_uc = logret[a].corr(logret[b])

        _shade_crisis_windows(ax, crisis_windows, annotate=ax is axes[0], data_end=data_end)
        ax.plot(logret.index, rho_t.values, color="black", lw=0.9)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(rho_uc, color="tab:red", lw=0.8, ls=":")
        ax.set_ylim(-1, 1)
        ax.set_ylabel(
            f"{_asset_label(a)}\n{_asset_label(b)}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.text(
            0.01,
            0.08,
            fr"$\rho={rho_uc:.2f}$",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:red",
        )
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")

    if show_title:
        if title is None:
            title = f"Rolling pairwise correlations, window = {window}"
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_qq_normal(
    logret: pd.DataFrame,
    ncols: int = 2,
    title: str = "QQ-plots against the normal distribution",
    show_title: bool = False,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
):
    """QQ-plot of standardized returns against N(0, 1)."""
    n = logret.shape[1]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (5.0 * ncols, 4.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, logret.columns):
        r = logret[col].dropna().astype(float).values
        r_std = (r - r.mean()) / r.std(ddof=1)
        stats.probplot(r_std, dist="norm", plot=ax)

        ex_kurt = stats.kurtosis(r, fisher=True, bias=False)
        ax.set_title(f"{_asset_label(col)} (excess kurtosis = {ex_kurt:.2f})")
        ax.grid(alpha=0.25)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if show_title:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def stats_to_latex(
    stats_df: pd.DataFrame,
    caption: str = "Descriptive statistics of daily log returns.",
    label: str = "tab:desc_stats",
    float_fmt: str = "%.4f",
) -> str:
    """Booktabs-style LaTeX table split into moments and diagnostics panels."""
    df = stats_df.copy()

    if "N" in df.columns:
        df["N"] = df["N"].astype(int)

    latex_names = {key: value.replace("&", "\\&") for key, value in ASSET_LABELS.items()}
    df.index = [latex_names.get(str(idx), str(idx)) for idx in df.index]

    def format_value(column: str, value) -> str:
        if pd.isna(value):
            return "--"
        if column == "N":
            return str(int(value))
        if column.endswith(" p") or column == "JB p" or column == "ADF p":
            value = float(value)
            if value < 1e-4:
                return "$<0.0001$"
            return float_fmt % value
        return float_fmt % float(value)

    moment_cols = ["N", "Mean", "SD", "Skew", "Kurtosis", "Min", "Max"]
    diagnostic_cols = [col for col in df.columns if col not in moment_cols]

    moment_header = ["", "N", "Mean", "SD", "Skew", "Kurt.", "Min", "Max"]
    diagnostic_header = [
        "",
        "JB",
        "$p_{JB}$",
        "$LB_{10}(r_t)$",
        "$p$",
        "$LB_{10}(r_t^2)$",
        "$p$",
        "ADF",
        "$p_{ADF}$",
    ]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "\\multicolumn{8}{l}{Panel A: Distributional moments} \\\\",
        "\\midrule",
        " & ".join(moment_header) + " \\\\",
        "\\midrule",
    ]

    for idx, row in df.iterrows():
        values = [str(idx)]
        values.extend(format_value(col, row[col]) for col in moment_cols)
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{0.35em}",
            "",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "\\multicolumn{9}{l}{Panel B: Diagnostic tests} \\\\",
            "\\midrule",
            " & ".join(diagnostic_header) + " \\\\",
            "\\midrule",
        ]
    )

    for idx, row in df.iterrows():
        values = [str(idx)]
        values.extend(format_value(col, row[col]) for col in diagnostic_cols)
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"
