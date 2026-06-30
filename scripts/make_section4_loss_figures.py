"""
Generate Section 4 training-loss figures from saved model histories.

Example:
    python scripts/make_section4_loss_figures.py
    python scripts/make_section4_loss_figures.py --run-id asof_2026-05-03_seed0
    python scripts/make_section4_loss_figures.py --run-prefix asof_2026-05-03_seed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "neural_bekk_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from section3_eda import set_thesis_style
from model_names import MODEL_ORDER, is_legacy_alias, model_label


@dataclass(frozen=True)
class LossHistory:
    run_id: str
    model_name: str
    seed: int | None
    train_nll: np.ndarray
    val_nll: np.ndarray
    best_val_nll: float | None
    early_stop_epoch: int | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Section 4 training-loss figures from results/runs histories."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/runs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional single run folder under --runs-root. Defaults to all runs.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="asof_2026-05-03_seed",
        help="Run-folder prefix used when --run-id is omitted.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures/section4"),
        help="Output directory for main-text figures.",
    )
    parser.add_argument(
        "--appendix-dir",
        type=Path,
        default=Path("figures/appendix"),
        help="Output directory for appendix figures.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="pdf",
        help="Figure output format.",
    )
    parser.add_argument(
        "--no-appendix",
        action="store_true",
        help="Do not write single-run appendix figures when plotting all runs.",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Add matplotlib titles. Usually leave off for thesis figures with LaTeX captions.",
    )
    return parser


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _history_paths(
    runs_root: Path, run_id: str | None, run_prefix: str
) -> list[Path]:
    if run_id is not None:
        return sorted((runs_root / run_id / "models").glob("*/history.json"))
    return sorted(runs_root.glob(f"{run_prefix}*/models/*/history.json"))


def _parse_seed(run_id: str, metadata_path: Path) -> int | None:
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        seed = metadata.get("seed")
        if seed is not None:
            return int(seed)

    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", run_id)
    if match:
        return int(match.group(1))
    return None


def _as_float_array(values: object, path: Path, key: str) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError(f"Expected '{key}' to be a list in {path}")
    return np.asarray(values, dtype=float)


def _read_history(path: Path) -> LossHistory:
    history = _read_json(path)
    train_nll = _as_float_array(history.get("train_epoch_nll"), path, "train_epoch_nll")
    val_nll = _as_float_array(history.get("val_epoch_nll"), path, "val_epoch_nll")
    if train_nll.shape != val_nll.shape:
        raise ValueError(f"Train and validation losses have different lengths in {path}")

    model_dir = path.parent
    run_dir = path.parents[2]
    metadata_path = model_dir / "run_metadata.json"

    best_val = history.get("best_val_nll")
    early_stop_epoch = history.get("early_stop_epoch")
    return LossHistory(
        run_id=run_dir.name,
        model_name=model_dir.name,
        seed=_parse_seed(run_dir.name, metadata_path),
        train_nll=train_nll,
        val_nll=val_nll,
        best_val_nll=None if best_val is None else float(best_val),
        early_stop_epoch=None if early_stop_epoch is None else int(early_stop_epoch),
    )


def _load_histories(
    runs_root: Path, run_id: str | None, run_prefix: str
) -> list[LossHistory]:
    paths = _history_paths(runs_root, run_id, run_prefix)
    if not paths:
        target = runs_root / run_id if run_id is not None else runs_root
        raise FileNotFoundError(f"No history.json files found below {target}")
    # Skip stale legacy-alias directories (duplicates of the canonical bekk_lstm_* runs).
    return [_read_history(path) for path in paths
            if not is_legacy_alias(path.parent.name)]


def _model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model_name), model_name
    except ValueError:
        return len(MODEL_ORDER), model_name


def _group_by_model(histories: list[LossHistory]) -> dict[str, list[LossHistory]]:
    grouped: dict[str, list[LossHistory]] = {}
    for history in histories:
        grouped.setdefault(history.model_name, []).append(history)
    return dict(sorted(grouped.items(), key=lambda item: _model_sort_key(item[0])))


def _group_by_run(histories: list[LossHistory]) -> dict[str, list[LossHistory]]:
    grouped: dict[str, list[LossHistory]] = {}
    for history in histories:
        grouped.setdefault(history.run_id, []).append(history)
    return dict(sorted(grouped.items()))


def _save_paths(base: Path, stem: str, output_format: str) -> list[Path]:
    if output_format == "both":
        return [base / f"{stem}.pdf", base / f"{stem}.png"]
    return [base / f"{stem}.{output_format}"]


def _series_matrix(histories: list[LossHistory], attr: str) -> np.ndarray:
    max_len = max(len(getattr(history, attr)) for history in histories)
    matrix = np.full((len(histories), max_len), np.nan)
    for i, history in enumerate(histories):
        values = getattr(history, attr)
        matrix[i, : len(values)] = values
    return matrix


def _panel_layout(n_panels: int) -> tuple[int, int]:
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def plot_loss_panels(
    histories: list[LossHistory],
    aggregate: bool,
    show_title: bool = False,
    save_path: str | Path | None = None,
):
    """Plot train and validation NLL, one panel per model."""
    grouped = _group_by_model(histories)
    n_panels = len(grouped)
    nrows, ncols = _panel_layout(n_panels)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.5, 2.25 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    train_color = "black"
    val_color = "tab:red"

    for ax, (model_name, model_histories) in zip(axes_flat, grouped.items()):
        model_histories = sorted(
            model_histories,
            key=lambda history: (-1 if history.seed is None else history.seed, history.run_id),
        )
        label = model_label(model_name)

        if aggregate and len(model_histories) > 1:
            for history in model_histories:
                epochs = np.arange(1, len(history.train_nll) + 1)
                ax.plot(epochs, history.train_nll, color=train_color, lw=0.45, alpha=0.18)
                ax.plot(epochs, history.val_nll, color=val_color, lw=0.45, alpha=0.18)

            train_matrix = _series_matrix(model_histories, "train_nll")
            val_matrix = _series_matrix(model_histories, "val_nll")
            epochs = np.arange(1, train_matrix.shape[1] + 1)
            ax.plot(
                epochs,
                np.nanmedian(train_matrix, axis=0),
                color=train_color,
                lw=1.5,
                label="Train NLL median",
            )
            ax.plot(
                epochs,
                np.nanmedian(val_matrix, axis=0),
                color=val_color,
                lw=1.5,
                ls="--",
                label="Validation NLL median",
            )
            ax.text(
                0.98,
                0.92,
                f"{len(model_histories)} seeds",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
        else:
            history = model_histories[0]
            epochs = np.arange(1, len(history.train_nll) + 1)
            ax.plot(epochs, history.train_nll, color=train_color, lw=1.3, label="Train NLL")
            ax.plot(epochs, history.val_nll, color=val_color, lw=1.3, ls="--", label="Validation NLL")
            if history.best_val_nll is not None:
                ax.text(
                    0.98,
                    0.92,
                    f"best val: {history.best_val_nll:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="0.35",
                )

        ax.set_title(label, loc="left", pad=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.grid(alpha=0.25)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    if show_title:
        if aggregate:
            fig.suptitle("Training and validation NLL across random seeds", y=0.995)
        else:
            run_ids = sorted({history.run_id for history in histories})
            fig.suptitle(f"Training and validation NLL ({', '.join(run_ids)})", y=0.995)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def main() -> None:
    matplotlib.use("Agg")
    args = build_arg_parser().parse_args()
    set_thesis_style()

    histories = _load_histories(args.runs_root, args.run_id, args.run_prefix)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.appendix_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id is None:
        for path in _save_paths(args.fig_dir, "training_losses_all_models", args.format):
            fig = plot_loss_panels(
                histories,
                aggregate=True,
                show_title=args.show_title,
                save_path=path,
            )
            plt.close(fig)

        if not args.no_appendix:
            for run_id, run_histories in _group_by_run(histories).items():
                for path in _save_paths(args.appendix_dir, f"training_losses_{run_id}", args.format):
                    fig = plot_loss_panels(
                        run_histories,
                        aggregate=False,
                        show_title=args.show_title,
                        save_path=path,
                    )
                    plt.close(fig)
    else:
        for path in _save_paths(args.fig_dir, f"training_losses_{args.run_id}", args.format):
            fig = plot_loss_panels(
                histories,
                aggregate=False,
                show_title=args.show_title,
                save_path=path,
            )
            plt.close(fig)

    model_names = ", ".join(_group_by_model(histories).keys())
    run_ids = ", ".join(_group_by_run(histories).keys())
    print(f"Loaded {len(histories)} histories.")
    print(f"Runs: {run_ids}")
    print(f"Models: {model_names}")
    print(f"Wrote main figures to: {args.fig_dir}")
    if args.run_id is None and not args.no_appendix:
        print(f"Wrote appendix figures to: {args.appendix_dir}")


if __name__ == "__main__":
    main()
