"""
Generate Section 3 EDA figures and the descriptive statistics table.

Example:
    python scripts/make_section3_figures.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "neural_bekk_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from section3_eda import (
    DEFAULT_CRISIS_WINDOWS,
    descriptive_stats_table,
    plot_log_returns,
    plot_qq_normal,
    plot_rolling_correlations,
    plot_squared_return_acf,
    set_thesis_style,
    stats_to_latex,
)


DEFAULT_MAIN_PAIRS = [
    ("^GSPC", "GC=F"),
    ("^GSPC", "^TNX"),
    ("GC=F", "CL=F"),
]


def _read_log_returns(path: Path, assets: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Date" not in df.columns:
        raise ValueError(f"Expected a 'Date' column in {path}")

    df = df.set_index("Date").sort_index()
    if assets is None:
        assets = list(df.columns)

    missing = [asset for asset in assets if asset not in df.columns]
    if missing:
        raise ValueError(f"Missing asset columns in {path}: {missing}")

    return df[assets].apply(pd.to_numeric, errors="coerce").dropna(how="all")


def _parse_assets(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_pairs(value: str | None) -> list[tuple[str, str]] | None:
    if value is None or value.strip() == "":
        return None

    pairs = []
    for raw_pair in value.split(","):
        parts = [item.strip() for item in raw_pair.split(":")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "Pairs must be comma-separated asset pairs like '^GSPC:GC=F,^GSPC:^TNX'."
            )
        pairs.append((parts[0], parts[1]))
    return pairs


def _filter_valid_pairs(
    requested_pairs: list[tuple[str, str]],
    columns: list[str],
) -> list[tuple[str, str]]:
    valid = []
    for a, b in requested_pairs:
        if a in columns and b in columns:
            valid.append((a, b))
    return valid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Section 3 EDA figures and descriptive statistics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/logret_2026-05-03.csv"),
        help="CSV file with Date plus return columns.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures/section3"),
        help="Output directory for main-text figures.",
    )
    parser.add_argument(
        "--appendix-dir",
        type=Path,
        default=Path("figures/appendix"),
        help="Output directory for appendix figures.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("tables"),
        help="Output directory for LaTeX tables.",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma-separated asset columns. Defaults to all columns except Date.",
    )
    parser.add_argument(
        "--main-pairs",
        type=str,
        default=None,
        help="Comma-separated asset pairs for the main rolling-correlation figure, e.g. '^GSPC:GC=F,^GSPC:^TNX'.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=120,
        help="Rolling-correlation window length.",
    )
    parser.add_argument(
        "--acf-lags",
        type=int,
        default=40,
        help="Number of ACF lags for squared returns.",
    )
    parser.add_argument(
        "--lb-lags",
        type=int,
        default=10,
        help="Ljung-Box lag used in the descriptive statistics table.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale applied to returns for plots and descriptive magnitudes.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="pdf",
        help="Figure output format.",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Add matplotlib titles. Usually leave off for thesis figures with LaTeX captions.",
    )
    parser.add_argument(
        "--no-shade-crises",
        action="store_false",
        dest="shade_crises",
        help="Disable shaded crisis windows in time-series figures.",
    )
    parser.set_defaults(shade_crises=True)
    return parser


def _save_paths(base: Path, stem: str, output_format: str) -> list[Path]:
    if output_format == "both":
        return [base / f"{stem}.pdf", base / f"{stem}.png"]
    return [base / f"{stem}.{output_format}"]


def main() -> None:
    matplotlib.use("Agg")
    args = build_arg_parser().parse_args()
    set_thesis_style()

    logret = _read_log_returns(args.input, _parse_assets(args.assets))
    crisis_windows = DEFAULT_CRISIS_WINDOWS if args.shade_crises else None

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.appendix_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    stats_df = descriptive_stats_table(logret, scale=args.scale, lb_lags=args.lb_lags)
    stats_df.to_csv(args.table_dir / "descriptive_statistics.csv")
    latex = stats_to_latex(
        stats_df,
        caption="Descriptive statistics of daily log returns. Mean, standard deviation, minimum, and maximum are reported in percent.",
        label="tab:desc_stats",
    )
    (args.table_dir / "descriptive_statistics.tex").write_text(latex, encoding="utf-8")

    for path in _save_paths(args.fig_dir, "log_returns", args.format):
        fig = plot_log_returns(
            logret,
            scale=args.scale,
            show_title=args.show_title,
            crisis_windows=crisis_windows,
            save_path=path,
        )
        plt.close(fig)

    for path in _save_paths(args.fig_dir, "squared_return_acf", args.format):
        fig = plot_squared_return_acf(
            logret,
            lags=args.acf_lags,
            show_title=args.show_title,
            save_path=path,
        )
        plt.close(fig)

    main_pairs = _parse_pairs(args.main_pairs)
    if main_pairs is None:
        main_pairs = _filter_valid_pairs(DEFAULT_MAIN_PAIRS, list(logret.columns))
        if not main_pairs:
            cols = list(logret.columns)
            main_pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))][:3]

    for path in _save_paths(args.fig_dir, "rolling_correlations_main", args.format):
        fig = plot_rolling_correlations(
            logret,
            window=args.rolling_window,
            pairs=main_pairs,
            show_title=args.show_title,
            crisis_windows=crisis_windows,
            save_path=path,
        )
        plt.close(fig)

    for path in _save_paths(args.appendix_dir, "rolling_correlations_all", args.format):
        fig = plot_rolling_correlations(
            logret,
            window=args.rolling_window,
            pairs=None,
            show_title=args.show_title,
            crisis_windows=crisis_windows,
            save_path=path,
        )
        plt.close(fig)

    for path in _save_paths(args.appendix_dir, "qq_normal", args.format):
        fig = plot_qq_normal(
            logret,
            show_title=args.show_title,
            save_path=path,
        )
        plt.close(fig)

    print(f"Wrote main figures to: {args.fig_dir}")
    print(f"Wrote appendix figures to: {args.appendix_dir}")
    print(f"Wrote tables to: {args.table_dir}")
    print("Main rolling-correlation pairs:", ", ".join(f"{a}-{b}" for a, b in main_pairs))


if __name__ == "__main__":
    main()
