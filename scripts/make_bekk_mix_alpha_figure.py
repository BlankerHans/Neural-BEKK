"""Plot the convex-mixture gate path for a trained BEKK-LSTM mix model.

Run from the neural_bekk project root:
    python scripts/make_bekk_mix_alpha_figure.py --seed 42 --scope all
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "neural_bekk_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bekk_kernel import BEKKLSTM
from section3_eda import DEFAULT_CRISIS_WINDOWS, _shade_crisis_windows, set_thesis_style
from vech import vech


DEFAULT_ASOF = "2026-05-03"
DEFAULT_MODEL = "bekk_lstm_mix"
DEFAULT_SCOPE = "all"
SPLITS = ("train", "val", "test")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and plot the gate for a trained convex-mixture BEKK-LSTM."
    )
    parser.add_argument("--asof", default=DEFAULT_ASOF, help="Data/run id, e.g. 2026-05-03.")
    parser.add_argument("--seed", type=int, default=42, help="Seed run to load.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model directory name.")
    parser.add_argument(
        "--scope",
        choices=("all", "test"),
        default=DEFAULT_SCOPE,
        help="Plot all train/validation/test forecasts or only test forecasts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "section4",
        help="Directory for the output figure.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "tables",
        help="Directory for alpha_t summary tables.",
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
        help="Add a matplotlib title. Usually leave off for thesis figures with captions.",
    )
    return parser


def add_cross_returns(df_returns: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    returns = df_returns[assets].to_numpy(dtype=np.float32)
    outer = returns[:, :, None] * returns[:, None, :]
    cross = vech(torch.from_numpy(outer)).numpy()
    cross_cols = [f"cross_{i + 1}" for i in range(cross.shape[1])]

    out = df_returns[assets].copy()
    out[cross_cols] = cross
    return out


def load_config(run_dir: Path, model_name: str) -> dict:
    config_path = run_dir / "models" / model_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_class") != "BEKKLSTM":
        raise ValueError(f"Expected BEKKLSTM config, got {config.get('model_class')!r}")
    if config.get("modulation") != "convex_mixture":
        raise ValueError(f"Expected convex_mixture modulation, got {config.get('modulation')!r}")
    return config


def load_splits(asof: str, n_assets: int):
    raw = {}
    for split in SPLITS:
        path = ROOT / "data" / f"{split}_df_{asof}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split data: {path}")
        raw[split] = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()

    assets = list(raw["train"].columns[:n_assets])
    if len(assets) != n_assets:
        raise ValueError(f"Expected {n_assets} asset columns, found {len(assets)}")

    feats = {split: add_cross_returns(df, assets) for split, df in raw.items()}
    mu = feats["train"].mean()
    sigma = feats["train"].std()
    norm = {split: (df - mu) / sigma for split, df in feats.items()}

    train_cov = torch.from_numpy(np.cov(norm["train"][assets].to_numpy(dtype=np.float32).T).astype(np.float32))
    return_std = torch.from_numpy(sigma[assets].to_numpy(dtype=np.float32))
    return raw, norm, assets, sigma, train_cov, return_std


def make_sequences(norm_df: pd.DataFrame, lookback: int) -> torch.Tensor:
    values = norm_df.to_numpy(dtype=np.float32)
    if len(values) <= lookback:
        raise ValueError(f"Need more than lookback={lookback} rows, got {len(values)}")
    sequences = np.stack([values[i:i + lookback] for i in range(len(values) - lookback)])
    return torch.from_numpy(sequences)


def build_model(config: dict, train_cov: torch.Tensor, return_std: torch.Tensor) -> BEKKLSTM:
    valid = set(inspect.signature(BEKKLSTM.__init__).parameters) - {"self"}
    kwargs = {key: value for key, value in config.items() if key in valid}
    kwargs["Sigma0"] = train_cov
    kwargs["return_std"] = return_std
    return BEKKLSTM(**kwargs)


def load_model(run_dir: Path, model_name: str, config: dict, train_cov: torch.Tensor, return_std: torch.Tensor) -> BEKKLSTM:
    checkpoint_path = run_dir / "models" / model_name / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = build_model(config, train_cov, return_std)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_last_alpha(model: BEKKLSTM, sequences: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        _, _, params = model(sequences, return_alpha=True)
    return params["alpha"][:, -1].cpu().numpy()


def extract_alpha_frame(model: BEKKLSTM, raw: dict, norm: dict, lookback: int, scope: str) -> pd.DataFrame:
    rows = []
    splits = ("test",) if scope == "test" else SPLITS
    for split in splits:
        sequences = make_sequences(norm[split], lookback)
        alpha = extract_last_alpha(model, sequences)
        dates = raw[split].index[lookback:]
        rows.append(pd.DataFrame({"Date": dates, "split": split, "alpha": alpha}))
    return pd.concat(rows, ignore_index=True)


def clipped_crisis_windows(data_start: pd.Timestamp, data_end: pd.Timestamp):
    clipped = []
    for label, start, end in DEFAULT_CRISIS_WINDOWS:
        start_ts = max(pd.Timestamp(start), data_start)
        end_ts = pd.Timestamp(end) if end is not None else data_end
        end_ts = min(end_ts, data_end)
        if start_ts <= end_ts:
            clipped.append((label, str(start_ts.date()), str(end_ts.date())))
    return clipped


def output_paths(out_dir: Path, stem: str, output_format: str) -> list[Path]:
    if output_format == "both":
        return [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    return [out_dir / f"{stem}.{output_format}"]


def plot_alpha(alpha_df: pd.DataFrame, raw: dict, scope: str, show_title: bool, save_path: Path) -> None:
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 2.6))
    data_start = pd.Timestamp(alpha_df["Date"].min())
    data_end = pd.Timestamp(alpha_df["Date"].max())

    _shade_crisis_windows(
        ax,
        clipped_crisis_windows(data_start, data_end),
        annotate=True,
        data_end=data_end,
    )

    for split in (("test",) if scope == "test" else SPLITS):
        split_df = alpha_df[alpha_df["split"] == split]
        ax.plot(split_df["Date"], split_df["alpha"], color="black", lw=0.9)

    if scope == "all":
        for label, split in (("Val.", "val"), ("Test", "test")):
            boundary = pd.Timestamp(raw[split].index.min())
            ax.axvline(boundary, color="0.35", lw=0.7, ls="--", alpha=0.7)
            ax.text(
                boundary,
                0.86,
                label,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7,
                color="0.30",
            )

    mean_alpha = float(alpha_df["alpha"].mean())
    ax.axhline(mean_alpha, color="tab:red", lw=0.8, ls=":")
    ax.text(
        0.01,
        0.08,
        fr"$\bar{{\lambda}}={mean_alpha:.2f}$",
        transform=ax.transAxes,
        fontsize=8,
        color="tab:red",
    )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"$\lambda_t$", rotation=0, ha="right", va="center", fontsize=9)
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    if show_title:
        ax.set_title("BEKK-LSTM convex-mixture gate")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def alpha_describe_table(alpha_df: pd.DataFrame) -> pd.DataFrame:
    percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    columns = {"Overall": alpha_df["alpha"].describe(percentiles=percentiles)}

    for split in SPLITS:
        split_alpha = alpha_df.loc[alpha_df["split"] == split, "alpha"]
        if not split_alpha.empty:
            columns[split.capitalize()] = split_alpha.describe(percentiles=percentiles)

    table = pd.DataFrame(columns)
    table = table.rename(
        index={
            "count": "N",
            "mean": "Mean",
            "std": "SD",
            "min": "Min",
            "5%": "5\\%",
            "25%": "25\\%",
            "50%": "Median",
            "75%": "75\\%",
            "95%": "95\\%",
            "max": "Max",
        }
    )
    table.index.name = "Statistic"
    return table


def export_alpha_describe_table(alpha_df: pd.DataFrame, table_dir: Path, stem: str) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    table = alpha_describe_table(alpha_df)
    csv_path = table_dir / f"{stem}_describe.csv"
    tex_path = table_dir / f"{stem}_describe.tex"

    table.to_csv(csv_path, float_format="%.6f")
    latex_table = table.rename_axis(None)
    latex_table.to_latex(
        tex_path,
        float_format="%.4f",
        escape=False,
        caption=r"Descriptive statistics of the BEKK-LSTM convex-mixture gate $\lambda_t$.",
        label=f"tab:{stem.replace('-', '_')}_describe",
    )
    return csv_path, tex_path


def main() -> None:
    args = build_arg_parser().parse_args()
    run_id = f"asof_{args.asof}_seed{args.seed}"
    run_dir = ROOT / "results" / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    config = load_config(run_dir, args.model)
    lookback = int(config.get("lookback", 40))
    raw, norm, _, _, train_cov, return_std = load_splits(args.asof, int(config["n_assets"]))
    model = load_model(run_dir, args.model, config, train_cov, return_std)
    alpha_df = extract_alpha_frame(model, raw, norm, lookback, args.scope)

    stem = f"{args.model}_alpha_{args.scope}_{args.asof}_seed{args.seed}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{args.model}_alpha_{args.scope}.csv"
    alpha_df.to_csv(csv_path, index=False)

    for path in output_paths(args.out_dir, stem, args.format):
        plot_alpha(alpha_df, raw, args.scope, args.show_title, path)

    describe_csv_path, describe_tex_path = export_alpha_describe_table(
        alpha_df,
        args.table_dir,
        stem,
    )

    print(f"Wrote alpha values to: {csv_path}")
    print(f"Wrote figure(s) to: {args.out_dir}")
    print(f"Wrote alpha describe table to: {describe_csv_path}")
    print(f"Wrote alpha LaTeX table to: {describe_tex_path}")
    print(
        "alpha summary:",
        f"mean={alpha_df['alpha'].mean():.4f}",
        f"min={alpha_df['alpha'].min():.4f}",
        f"max={alpha_df['alpha'].max():.4f}",
    )


if __name__ == "__main__":
    main()
