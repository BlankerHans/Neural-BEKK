# Reproducing the Thesis Results

This document describes the order in which the repository files are used to
reproduce the empirical results. Run all commands from the repository root.

There are two distinct reproduction targets:

1. **Artifact-level reproduction (recommended):** rebuild the reported tables
   and figures from the committed checkpoints, forecasts, and loss matrices.
   This is fast and reproduces the submitted thesis results.
2. **Full re-estimation:** download or rebuild the data, retrain every neural
   model, refit the R benchmarks, and rerun every backtest. This is expensive
   and will not be byte-identical because neural training, hardware, package
   versions, and upstream market-data revisions can affect the estimates.

## 1. Relevant Files and Directories

| Path | Role in the workflow |
|---|---|
| `data/` | Versioned return data and chronological train/validation/test splits |
| `run_models.ipynb` | Main four-asset pipeline: data, neural training, R benchmarks, and backtests |
| `run_models_code_only.py` | Exported notebook code for inspection; the notebook is the primary driver |
| `LSTMCovariance.py`, `GRUCovariance.py` | Direct neural covariance benchmarks |
| `bekk_kernel.py` | Scalar, vector, and convex-mixture BEKK-LSTM models |
| `neural_bekk.py` | Diagonal and full time-varying Neural-BEKK models |
| `training_functions.py` | Likelihoods, optimization, early stopping, and checkpoint persistence |
| `r/bekks.R` | Symmetric and asymmetric BEKK estimation |
| `r/dcc.R` | DCC and ADCC estimation |
| `backtesting.py`, `risk_metrics.py` | VaR/ES construction, calibration tests, FZ0 loss, and DM tests |
| `results/runs/` | Original per-seed checkpoints and main backtest artifacts |
| `results/runs_with_full/` | Final main-sample loss/backtest artifacts including full Neural-BEKK |
| `results/evaluation_with_full/` | Final aggregate tables and figures reported in the thesis |
| `evaluate_runs.py` | Cross-seed aggregation and 90% model confidence sets |
| `scripts/` | Follow-up training, robustness checks, and figure generation |

The main experiment is identified by `DATA_ID=2026-05-03` and uses the seeds
`42, 0, 1, ..., 9`. Keep this identifier when comparing results with the
submitted tables. Files tagged `gspc_tnx_1962` and `gspc_tnx_1990` belong to the
two-asset robustness study.

## 2. Environment Setup

Create the Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the R packages needed by the econometric models and the R-backed
backtests:

```r
install.packages(c("BEKKs", "rugarch", "rmgarch", "esback", "segMGarch", "MCS"))
```

`Rscript` must be available on `PATH`, and `rpy2` must be able to load the same
R installation. Verify the Python code before reproducing results:

```bash
python -m unittest discover -s tests -v
```

## 3. Recommended: Rebuild the Submitted Results from Saved Artifacts

The committed `results/runs_with_full/` directory contains the final
per-seed FZ0 loss matrices and backtest results for all 13 specifications,
including the full asymmetric Neural-BEKK. Rebuild the cross-seed evaluation in
a separate output directory:

```bash
python evaluate_runs.py \
  --runs-dir results/runs_with_full \
  --out results/reproduced/main \
  --no-overleaf
```

The regenerated main table is written to:

```text
results/reproduced/main/asof_2026-05-03/table1_main.csv
```

Compare it with the submitted result:

```bash
diff -u \
  results/evaluation_with_full/asof_2026-05-03/table1_main.csv \
  results/reproduced/main/asof_2026-05-03/table1_main.csv
```

The same evaluation directory also contains:

- `table2_var_backtests.csv`: VaR calibration tests;
- `table3_es_backtests.csv`: Expected Shortfall tests;
- `table4_dm_benchmark.csv`: DM comparisons against symmetric BEKK;
- `per_seed_pvalues.csv`: all seed-level test results;
- `fig_fz_loss_seeds.pdf`: FZ0 loss distribution across seeds;
- `fig_cum_loss_diff.pdf`: cumulative loss differential;
- `report.md`: compact human-readable summary.

This route does not retrain models and does not require downloading data.

## 4. Regenerate Figures from Existing Checkpoints

The following scripts use the committed data and model artifacts:

```bash
# Data description, correlations, and squared-return ACFs
python scripts/make_section3_figures.py

# Training and validation loss paths across seeds
python scripts/make_section4_loss_figures.py

# FHS VaR paths for all models (seed 42)
python scripts/make_fhs_var_allmodels.py

# Convex-mixture gate and its summary table
python scripts/make_bekk_mix_alpha_figure.py --seed 42 --scope all
```

The outputs are written to `figures/` and `tables/`. These commands may replace
files already committed there; use the scripts' `--fig-dir`, `--out-dir`, or
`--table-dir` options when a separate comparison directory is preferred.

## 5. Reproduce the Robustness Evaluations

### QLIKE

QLIKE reconstructs covariance forecasts from the saved Python checkpoints and
R forecasts; it does not retrain the models.

```bash
QLIKE_TAG=2026-05-03 python scripts/qlike_mcs.py
```

Results are written to `results/evaluation_qlike/2026-05-03/`. The two-asset
cohorts can be evaluated with:

```bash
QLIKE_TAG=gspc_tnx_1990 python scripts/qlike_mcs.py
QLIKE_TAG=gspc_tnx_1962 python scripts/qlike_mcs.py
```

### Alternative FHS Tail Level

The committed 5% FHS results can be re-aggregated without recomputing model
forecasts:

```bash
python evaluate_runs.py \
  --runs-dir results/runs_alpha05 \
  --out results/reproduced/alpha05 \
  --no-overleaf
```

To recompute the 5% risk forecasts and backtests from the saved checkpoints:

```bash
python scripts/backtest_fhs_alpha05.py
```

### Parametric VaR and ES

```bash
python scripts/parametric_var_es.py
python evaluate_runs.py \
  --runs-dir results/runs_parametric \
  --out results/reproduced/parametric \
  --no-overleaf
```

### Alternative Two-Asset Samples

The committed two-asset runs are stored beside the main runs in
`results/runs/`. Aggregate all stored cohorts with:

```bash
python evaluate_runs.py \
  --runs-dir results/runs \
  --out results/reproduced/cohorts \
  --no-overleaf
```

The relevant outputs are the `asof_gspc_tnx_1962/` and
`asof_gspc_tnx_1990/` subdirectories.

## 6. Full Re-estimation

Full re-estimation is only necessary when the model fits themselves must be
reproduced. It is substantially slower than artifact-level reproduction.

### 6.1 Make the Drivers Portable

Before running the pipeline from a different clone location:

1. replace the absolute `project_dir` values in `run_models.ipynb`;
2. replace the hard-coded `cd` target in `scripts/run_all_seeds.sh`;
3. confirm that `Rscript` and the Python environment are available in the same
   shell.

The notebook uses the fixed as-of date `2026-05-03`, writes the data splits to
`data/`, R forecasts to `r/output/`, and each seed to
`results/runs/asof_2026-05-03_seed<SEED>/`.

### 6.2 Run One Seed as a Smoke Test

```bash
MODEL_SEED=42 jupyter nbconvert \
  --to notebook \
  --execute run_models.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --stdout > /dev/null
```

Verify that the run directory contains:

```text
results/runs/asof_2026-05-03_seed42/
|-- models/<model>/checkpoint.pt
|-- models/<model>/config.json
|-- models/<model>/history.json
|-- all_backtest_results.json
|-- fz_loss_matrix.csv
`-- fz_loss_summary.csv
```

### 6.3 Train the Eleven Main Seeds

```bash
bash scripts/run_all_seeds.sh
```

The script executes the notebook for seeds `42, 0, 1, ..., 9`. Each notebook
run trains the direct LSTM/GRU models, the three BEKK-LSTM variants, and the
diagonal Neural-BEKK; it also fits the BEKK/DCC benchmarks and performs the
main FHS backtests.

### 6.4 Train the Full Neural-BEKK Follow-up

The full asymmetric Neural-BEKK was trained separately after the main sweep:

```bash
python -u scripts/train_neural_bekk_full.py
```

The script is resumable and skips seeds whose
`models/neural_bekk_asym_full/checkpoint.pt` already exists. To train only a
specific subset:

```bash
NBEKK_SEEDS="42 0" python -u scripts/train_neural_bekk_full.py
```

### 6.5 Recompute a Fresh Main Evaluation Including the Full Model

```bash
FHS_ALPHA=0.01 python scripts/backtest_fhs_alpha05.py

python evaluate_runs.py \
  --runs-dir results/runs_alpha01 \
  --out results/reproduced/fresh_full \
  --no-overleaf
```

This produces a fresh, internally consistent evaluation of every checkpoint
with the current common FHS reconstruction code. It is a computational
replication, not the canonical submitted table. For the exact thesis table,
use the artifact-level workflow in section 3, which preserves the original
per-model forecast paths in `results/runs_with_full/`.

## 7. Reproduction Checklist

A successful artifact-level reproduction should satisfy all of the following:

- the unit test suite passes;
- all eleven seed directories are discovered;
- the main evaluation contains 13 model specifications;
- every model has 925 aligned out-of-sample observations;
- `table1_main.csv` is headed by the diagonal Neural-BEKK, BEKK-LSTM mixture,
  and scalar BEKK-LSTM under mean FZ0 loss;
- the submitted and regenerated main CSV tables agree apart from any explicitly
  documented software-version formatting differences.

For exact comparisons, do not mix the main `2026-05-03` files with the
`gspc_tnx_*` robustness cohorts, and do not overwrite the committed artifacts
before creating a separate reproduction output directory.
