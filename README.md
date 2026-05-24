# Neural BEKK

This repository contains the research and experiment code for a master's thesis
on multivariate volatility, covariance, and tail-risk forecasting. The central
idea is to combine neural sequence models with BEKK-style covariance recursions
and compare them with classical econometric GARCH benchmarks.

The project is structured as a reproducible empirical research workspace rather
than as a packaged Python library. Data preparation, model training, benchmark
estimation, backtesting, figures, tables, and saved model artifacts are kept
together so that the empirical pipeline remains inspectable.

## Research Goal

The thesis asks whether neural covariance models can capture the dynamics of
multivariate financial time series better than purely parametric GARCH
benchmarks. The empirical application uses daily returns for:

- S&P 500 (`^GSPC`)
- Gold Futures (`GC=F`)
- Crude Oil Futures (`CL=F`)
- 10Y Treasury Yield (`^TNX`)

Each model produces a positive semidefinite conditional covariance forecast.
These forecasts are then used for portfolio volatility, Value-at-Risk, Expected
Shortfall, and comparative forecast evaluation.

## Implemented Models

The repository implements three groups of models.

**1. Direct neural covariance models**

`LSTMCovariance` and `GRUCovariance` map a rolling input window to a Cholesky
factor of the next covariance matrix:

$$h_t = \mathrm{RNN}_{\theta}(x_{t-L+1:t}), \quad \ell_t = W h_t + b.$$

$$L_t = \mathrm{lower}(\ell_t), \quad \mathrm{diag}(L_t) = \mathrm{softplus}(\mathrm{diag}(L_t)) + \varepsilon.$$

$$\widehat{\Sigma}_{t+1} = L_t L_t^\top.$$

This construction enforces positive semidefiniteness by design.

**2. Neural BEKK with time-varying parameters**

`NeuralBekk` uses a GRU to generate BEKK parameters at each step. In the
diagonal variant, the implemented recursion is:

$$\widehat{\Sigma}_{t+1} = C_t C_t^\top + (a_t \odot \epsilon_t)(a_t \odot \epsilon_t)^\top + \widehat{\Sigma}_t \odot (g_t g_t^\top) + I_{\mathrm{asym}}(b_t \odot \eta_t)(b_t \odot \eta_t)^\top.$$

where

$$\eta_t = \min(\epsilon_t, 0).$$

The full-matrix variant uses the same BEKK logic with matrix-valued
coefficients:

$$\widehat{\Sigma}_{t+1} = C_t C_t^\top + A_t \epsilon_t \epsilon_t^\top A_t^\top + G_t \widehat{\Sigma}_t G_t^\top + I_{\mathrm{asym}} B_t \eta_t \eta_t^\top B_t^\top.$$

The implementation constrains persistence through bounded coefficient scales so
that the covariance recursion remains numerically stable.

**3. BEKK-LSTM kernel models**

`BEKKLSTM` embeds a BEKK kernel inside a recurrent cell. The base covariance
kernel is:

$$K_{t+1} = C C^\top + A^\top \epsilon_t \epsilon_t^\top A + B^\top \Sigma_t B + I_{\mathrm{asym}} G^\top \eta_t \eta_t^\top G.$$

The recurrent hidden state then modulates this kernel. The implemented
modulation variants are:

Scalar modulation:

$$\Sigma_{t+1} = m_t K_{t+1}.$$

Vector modulation:

$$\Sigma_{t+1} = \mathrm{diag}(m_t) K_{t+1} \mathrm{diag}(m_t).$$

Convex mixture:

$$\Sigma_{t+1} = (1-\alpha_t)K_{t+1} + \alpha_t L^{NN}_t(L^{NN}_t)^\top.$$

Classical symmetric/asymmetric BEKK and DCC/aDCC-GARCH benchmarks are estimated
through the R scripts in `r/`.

## Evaluation

Model quality is evaluated with Gaussian and Student-t negative log-likelihood,
Filtered Historical Simulation VaR/ES, Kupiec and Christoffersen VaR tests,
dynamic quantile tests, ES backtests, Fissler-Ziegel loss, and
Diebold-Mariano-style forecast comparisons.

## Repository Structure

```text
.
|-- neural_bekk.py              # Neural-BEKK model with time-varying parameters
|-- bekk_kernel.py              # BEKK-LSTM cell and BEKK-based recurrence
|-- LSTMCovariance.py           # LSTM baseline for covariance forecasts
|-- GRUCovariance.py            # GRU baseline for covariance forecasts
|-- training_functions.py       # Training loops, loss functions, checkpoints
|-- backtesting.py              # VaR/ES backtests and forecast comparisons
|-- risk_metrics.py             # VaR/ES helper functions
|-- run_models.ipynb            # Main notebook for the empirical pipeline
|-- run_models_code_only.py     # Exported code version of the notebook pipeline
|-- r/
|   |-- bekks.R                 # Symmetric/asymmetric BEKK benchmarks
|   |-- dcc.R                   # DCC/aDCC benchmarks
|   `-- output/                 # Covariance forecasts produced by R
|-- scripts/                    # Figure generation and multi-seed runs
|-- data/                       # Versioned raw data and train/val/test splits
|-- results/                    # Model artifacts, checkpoints, backtests
|-- figures/                    # Thesis figures
|-- tables/                     # Descriptive tables
`-- tests/                      # Unit tests for model and backtesting logic
```

## Setup

Recommended Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy statsmodels matplotlib yfinance torch jupyter pytest
```

The R benchmarks and some ES/DQ backtests require additional R packages:

```r
install.packages(c("BEKKs", "rugarch", "rmgarch", "esback", "segMGarch"))
```

Optionally install `rpy2` if Python should call the R-based backtests directly.

## Usage

Run the unit tests with:

```bash
python -m unittest discover -s tests
```

Generate exploratory figures and descriptive tables from the repository root:

```bash
python scripts/make_section3_figures.py
```

Generate training-loss figures from saved model histories:

```bash
python scripts/make_section4_loss_figures.py
```

The main empirical run is stored in the notebook:

```bash
jupyter nbconvert \
  --to notebook \
  --execute run_models.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Multiple seeds can be executed with:

```bash
bash scripts/run_all_seeds.sh
```

Practical note: some experimental drivers were exported from a local notebook
workflow and contain absolute project paths. If the repository is cloned to a
different location, update `project_dir` in the relevant notebook or exported
script.

## Data and Reproducibility

Data are downloaded via `yfinance` and versioned with a `DATA_ID`. The data
snapshot currently used in the code is `2026-05-03`. Train, validation, and
test splits are stored as CSV files in `data/`.

Model runs are controlled through `MODEL_SEED`. Artifacts are written to:

```text
results/runs/asof_<DATA_ID>_seed<SEED>/
```

Each run directory contains model configurations, training histories,
checkpoints, and aggregated backtest results.

## Result Artifacts

The repository already contains generated intermediate and result files:

- `figures/`: main-text and appendix figures
- `tables/`: descriptive tables in CSV and LaTeX format
- `r/output/`: covariance forecasts from classical BEKK and DCC benchmarks
- `results/runs/`: trained models, histories, and backtest summaries

These artifacts make it possible to inspect results without retraining. For new
runs, use a separate `MODEL_SEED` or data snapshot to avoid overwriting existing
results.

## Scope

The code is optimized for empirical transparency in a scientific thesis. It
prioritizes explicit model variants, saved intermediate outputs, and
reproducible backtests over a polished package API. For production-style reuse,
the next practical steps would be an explicit dependency file, relative project
paths, and a single command-line entry point.
