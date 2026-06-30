"""Build the four-asset dataset with an approximated 10Y Treasury return."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from return_transforms import approximate_treasury_return_from_yield


ASSETS = ["^GSPC", "GC=F", "CL=F", "^TNX"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-close", default="data/close_2026-05-03.csv")
    parser.add_argument("--data-id", default="2026-05-03_treasury_return")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ROOT / args.source_close
    output_dir = ROOT / "data"

    close = pd.read_csv(source, index_col=0, parse_dates=True)[ASSETS].dropna()
    returns = np.log(close).diff()
    returns["^TNX"] = approximate_treasury_return_from_yield(close["^TNX"])
    returns = returns.dropna()

    train_end = int(len(returns) * args.train_size)
    val_end = int(len(returns) * (args.train_size + args.val_size))
    splits = {
        "train": returns.iloc[:train_end],
        "val": returns.iloc[train_end:val_end],
        "test": returns.iloc[val_end:],
    }

    close.to_csv(output_dir / f"close_{args.data_id}.csv", index_label="Date")
    returns.to_csv(output_dir / f"logret_{args.data_id}.csv", index_label="Date")
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}_df_{args.data_id}.csv", index_label="Date")

    print(f"data_id: {args.data_id}")
    print(f"sample: {returns.index.min().date()} -> {returns.index.max().date()} ({len(returns)})")
    print("standard deviations:")
    print(returns.std().to_string())
    for name, split in splits.items():
        print(f"{name}: {len(split)} ({split.index.min().date()} -> {split.index.max().date()})")


if __name__ == "__main__":
    main()
