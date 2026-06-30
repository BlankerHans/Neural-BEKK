"""Prepare a 2-asset (S&P 500 + 10Y Treasury yield) long-history dataset.

Robustness study: the main thesis dataset starts in 2000 because the gold/crude
*futures* (GC=F, CL=F) only begin then. Dropping the two commodities and keeping
only ^GSPC + ^TNX extends the sample back to ~1962, giving a far larger
out-of-sample window for the forecast-comparison tests (DM / MCS).

Returns are built exactly as in the main pipeline:
  - log returns for the price series (^GSPC),
  - duration/carry approximated Treasury returns from the yield series (^TNX).

Parameterized via environment variables so the same script produces both the
full-history cohort and a modern, more-stationary post-Volcker cohort:
    GSPC_DATA_ID      output namespace            (default gspc_tnx_1962)
    GSPC_START        keep observations >= this   (default 1900-01-01 -> ~1962)
    GSPC_TRAIN_SIZE   train fraction              (default 0.60)
    GSPC_VAL_SIZE     val fraction                (default 0.15; test = rest)
    GSPC_SOURCE_CLOSE reuse an existing close CSV instead of downloading
                      (e.g. data/close_gspc_tnx_1962.csv) -> guarantees the
                      modern cohort is an exact subset of the full-history data

Examples:
    # full history (default)
    python scripts/prepare_gspc_tnx_data.py
    # modern post-Volcker cohort, 50/15/35, reusing the already-downloaded close
    GSPC_DATA_ID=gspc_tnx_1990 GSPC_START=1990-01-01 \
      GSPC_TRAIN_SIZE=0.50 GSPC_VAL_SIZE=0.15 \
      GSPC_SOURCE_CLOSE=data/close_gspc_tnx_1962.csv \
      python scripts/prepare_gspc_tnx_data.py

Writes versioned split CSVs (same format/columns as the main pipeline) under
``data/`` with the chosen DATA_ID.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


DATA = ROOT / "data"

DATA_ID = os.environ.get("GSPC_DATA_ID", "gspc_tnx_1962")
START = os.environ.get("GSPC_START", "1900-01-01")
TRAIN_SIZE = float(os.environ.get("GSPC_TRAIN_SIZE", "0.60"))
VAL_SIZE = float(os.environ.get("GSPC_VAL_SIZE", "0.15"))
SOURCE_CLOSE = os.environ.get("GSPC_SOURCE_CLOSE", "")

TICKERS = ["^GSPC", "^TNX"]          # equity index + 10Y yield
END = "2026-05-02"                   # exclusive in yfinance; matches as-of 2026-05-01


def load_close() -> pd.DataFrame:
    """Either reuse an already-downloaded close CSV or pull from yfinance."""
    if SOURCE_CLOSE:
        src = (ROOT / SOURCE_CLOSE) if not os.path.isabs(SOURCE_CLOSE) else Path(SOURCE_CLOSE)
        close = pd.read_csv(src, index_col=0, parse_dates=True)[TICKERS]
    else:
        import yfinance as yf
        raw = yf.download(TICKERS, start="1900-01-01", end=END,
                          auto_adjust=False, progress=False)
        close = raw["Close"][TICKERS]
    close = close.dropna()
    return close[close.index >= pd.Timestamp(START)]


def train_val_test_split(df: pd.DataFrame, train_size: float, val_size: float):
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def main() -> None:
    close = load_close()

    logret = np.log(close).diff()
    logret["^TNX"] = close["^TNX"].diff() # r_t(yield) = y_t - y_{t-1}
    logret = logret.dropna()

    close.to_csv(DATA / f"close_{DATA_ID}.csv")
    logret.to_csv(DATA / f"logret_{DATA_ID}.csv")

    train, val, test = train_val_test_split(logret, TRAIN_SIZE, VAL_SIZE)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(DATA / f"{name}_df_{DATA_ID}.csv")

    print(f"tickers: {TICKERS}")
    print(f"sample:  {logret.index.min().date()} -> {logret.index.max().date()} "
          f"({len(logret)} obs)  split {TRAIN_SIZE:.2f}/{VAL_SIZE:.2f}")
    print(f"train:   {len(train)}  ({train.index.min().date()}..{train.index.max().date()})")
    print(f"val:     {len(val)}  ({val.index.min().date()}..{val.index.max().date()})")
    print(f"test:    {len(test)}  ({test.index.min().date()}..{test.index.max().date()})")
    print(f"DATA_ID: {DATA_ID}")


if __name__ == "__main__":
    main()
