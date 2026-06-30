"""Train the FULL (non-diagonal) asymmetric Neural-BEKK for every seed.

The thesis pipeline trains the *diagonal* Neural-BEKK (`neural_bekk_asym_diag`).
Matrix-QLIKE showed the hybrids' weakness is the off-diagonal / correlation
structure -> the full variant (full A/G/B matrices) adds exactly that. This
script trains ONLY that variant per seed and drops the checkpoint into a new
model dir, so the existing globbing evaluators (qlike_mcs.py, the FHS recompute)
pick it up automatically -- no monolith re-run, no retraining of the other 11.

Training mirrors the diagonal Neural-BEKK cell of run_models_code_only.py
exactly (same data via the validated load_splits reconstruction, same
hyperparameters, train_covariance_model + StudentTLoss); the ONLY change is
bekk_type="diag" -> "full". Saves to:
    results/runs/asof_<ASOF>_seed<k>/models/neural_bekk_asym_full/

Run in your own terminal (multi-hour, all seeds):
    python -u scripts/train_neural_bekk_full.py 2>&1 | tee /tmp/nbekk_full.log
Resumable: seeds whose model dir already exists are skipped.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gen_seq_data import SequenceDataset
from neural_bekk import NeuralBekk
from training_functions import train_covariance_model, save_model_checkpoint, StudentTLoss
from make_fhs_var_allmodels import load_splits, ASSETS, ASOF, LOOKBACK

RUNS = ROOT / "results" / "runs"
MODEL_NAME = "neural_bekk_asym_full"

# Hyperparameters copied verbatim from the diagonal Neural-BEKK cell.
N = len(ASSETS)
K = N * (N + 1) // 2
INPUT_SIZE = N + K
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.1
BATCHSIZE = 64
EPOCHS = 500
LR = 1e-4
BEKK_JITTER = 1e-4
PLATEAU_PATIENCE = 20
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 1e-4
DEVICE = "cpu"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def train_one(seed: int, raw, norm, sigma, train_cov, return_std, models_dir: Path) -> None:
    seed_everything(seed)
    train_ds = SequenceDataset(norm["train"].values, norm["train"][ASSETS].values, LOOKBACK)
    val_ds = SequenceDataset(norm["val"].values, norm["val"][ASSETS].values, LOOKBACK)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True,
                              generator=make_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False)

    loss_fn = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)
    model = NeuralBekk(
        n_assets=N, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=DROPOUT, bekk_type="full", asym=True, Sigma0=train_cov, return_std=return_std,
        bekk_scale=100.0, sigma0_in_bekk_scale=False, tau_max=0.995, c_delta_scale=0.01,
        init_C_from_sigma0=True, init_a=0.05, init_g=0.90, init_b=0.05, jitter=BEKK_JITTER,
    )
    model, history = train_covariance_model(
        model, train_loader, val_loader, loss_fn=loss_fn, loss_kwargs=None,
        epochs=EPOCHS, lr=LR, plateau_patience=PLATEAU_PATIENCE, device=DEVICE,
        scheduler_type="cosine", grad_clip_max_norm=0.5,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    config = {
        "model_class": "NeuralBekk", "distribution": "student_t", "bekk_type": "full",
        "asym": True, "input_size": INPUT_SIZE, "n_assets": N, "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS, "dropout": DROPOUT, "lookback": LOOKBACK,
        "batchsize": BATCHSIZE, "epochs": EPOCHS, "lr": LR, "scheduler_type": "cosine",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA, "device": DEVICE,
        "bekk_scale": 100.0, "jitter": BEKK_JITTER, "sigma0_in_bekk_scale": False,
        "tau_max": 0.995, "c_delta_scale": 0.01, "init_C_from_sigma0": True,
        "init_a": 0.05, "recursion_jitter": 0.0, "cholesky_jitter": BEKK_JITTER,
        "init_g": 0.90, "init_b": 0.05,
        "nu": float(loss_fn.nu.detach().cpu()),
    }
    save_model_checkpoint(
        models_dir=models_dir, model_name=MODEL_NAME, model=model, history=history,
        run_metadata={"trained_by": "scripts/train_neural_bekk_full.py", "seed": seed},
        loss_fn=loss_fn, config=config,
    )


def main() -> None:
    seed_dirs = sorted(RUNS.glob(f"asof_{ASOF}_seed*"))
    if not seed_dirs:
        raise SystemExit(f"no run dirs under {RUNS}/asof_{ASOF}_seed*")
    only = os.environ.get("NBEKK_SEEDS")  # e.g. "42" for a quick smoke test
    wanted = set(only.split()) if only else None

    # Data is identical across seeds -> build the reconstruction once.
    raw, norm, sigma, train_cov, return_std = load_splits()
    print(f"full asym Neural-BEKK | assets={ASSETS} (N={N}) | {len(seed_dirs)} seed dirs")

    for d in seed_dirs:
        seed = d.name.split("seed")[-1]
        if wanted is not None and seed not in wanted:
            continue
        out = d / "models" / MODEL_NAME / "checkpoint.pt"
        if out.exists():
            print(f"[skip] seed {seed}: {MODEL_NAME} already trained")
            continue
        print(f"\n=== seed {seed} -> {d.name}/models/{MODEL_NAME} ===", flush=True)
        train_one(int(seed), raw, norm, sigma, train_cov, return_std, d / "models")


if __name__ == "__main__":
    main()
