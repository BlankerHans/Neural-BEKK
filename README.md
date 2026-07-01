# Hybrid Neural-MGARCH Models for Volatility and Financial Risk Forecasting

This repository contains the implementation and replication artifacts for the
master's thesis *Hybrid Neural-MGARCH Models for Volatility and Financial Risk
Forecasting*. It studies whether recurrent neural networks can improve
multivariate covariance and portfolio tail-risk forecasts when they are
combined with the structure of a BEKK model.

The comparison is deliberately bidirectional. Hybrid models are evaluated
against both classical econometric benchmarks (BEKK, DCC, and ADCC) and purely
data-driven recurrent covariance models (LSTM and GRU). The repository is a
research workspace rather than a packaged Python library: data preparation,
training, evaluation, figures, tables, checkpoints, and R benchmark forecasts
are kept together to make the empirical pipeline inspectable.

The submitted master's thesis is available as
[`neural_bekk_masterthesis_dinkela.pdf`](neural_bekk_masterthesis_dinkela.pdf).

## Research Design

All models produce one-step-ahead forecasts of the conditional covariance
matrix

$$
\Sigma_t = \mathrm{Cov}(r_t \mid \Omega_{t-1}),
$$

where $r_t = \varepsilon_t \in \mathbb{R}^N$ is the zero-mean return vector and
$\Omega_{t-1}$ is the information available at the forecast origin. The main
experiment uses $N=4$ assets and compares three model classes:

| Class | Implemented models | Covariance mechanism |
|---|---|---|
| Econometric | symmetric/asymmetric BEKK, DCC, ADCC | fixed MGARCH recursions |
| Neural | LSTM and GRU under Gaussian and Student-t likelihoods | direct Cholesky output |
| Hybrid | scalar/vector/mixture BEKK-LSTM, diagonal/full Neural-BEKK | recurrent modulation or time-varying BEKK parameters |

The principal methodological contribution is a deterministic, asymmetric
Neural-BEKK whose diagonal or full parameter matrices vary with the information
set. The second hybrid family places a time-invariant asymmetric BEKK kernel
inside a custom LSTM-style recurrent cell.

## Notation and Information Set

The notation follows the thesis:

- $\mathrm{vec}(M)$ stacks the columns of a matrix.
- $\mathrm{vech}(M)$ stacks the lower-triangular part of a symmetric
  matrix, including its diagonal.
- $\mathrm{unvech}_{\mathrm{lt}}(v)$ fills a lower-triangular matrix from
  a vector without mirroring its off-diagonal entries.
- $\eta_t = \mathbf{1}\{\varepsilon_t<0\}\odot\varepsilon_t$ contains the
  component-wise negative innovations.

For a forecast at date $t$, every recurrent model receives

$$
x_{t-1} = \left(r_{t-1},\mathrm{vech}(r_{t-1}r_{t-1}^{\top})\right),
\qquad
X_t=(x_{t-s},\ldots,x_{t-1}),
$$

with lookback $s=40$. Thus $X_t\mapsto\widehat{\Sigma}_{t\mid t-1}$ uses only
information available through $t-1$. With four assets, each feature vector has
$4+4(4+1)/2=14$ entries. No exogenous variables enter the reported models.

## Model Specifications

### Classical MGARCH Benchmarks

The full symmetric BEKK(1,1) recursion is

$$
\Sigma_t
=CC^{\top}
+A^{\top}\varepsilon_{t-1}\varepsilon_{t-1}^{\top}A
+G^{\top}\Sigma_{t-1}G,
$$

where $C$ is lower triangular and $A,G\in\mathbb{R}^{N\times N}$. The
component-wise asymmetric extension used as the theoretical building block for
the hybrid models is

$$
\Sigma_t
=CC^{\top}
+A^{\top}\varepsilon_{t-1}\varepsilon_{t-1}^{\top}A
+G^{\top}\Sigma_{t-1}G
+B^{\top}\eta_{t-1}\eta_{t-1}^{\top}B.
$$

The quadratic forms preserve positive definiteness for a full-rank $C$. The R
benchmark in `r/bekks.R` follows the joint-sign asymmetry implemented by the
`BEKKs` package; this differs from the component-wise selector $\eta_t$ used by
the Python hybrid models. `r/dcc.R` estimates scalar DCC and ADCC models with
univariate GARCH(1,1) margins.

### Direct LSTM and GRU Covariance Models

The neural benchmarks map the final recurrent hidden state to the
$N(N+1)/2$ elements of a lower-triangular matrix:

$$
v_t=W_{\mathrm{out}}h_{t-1}+b_{\mathrm{out}},
\qquad
L_t=\mathrm{unvech}_{\mathrm{lt}}(v_t).
$$

The diagonal of $L_t$ is transformed with `softplus` and a small positive
jitter. The forecast

$$
\widehat{\Sigma}_{t\mid t-1}=L_tL_t^{\top}
$$

is therefore positive definite by construction. The LSTM and GRU variants are
implemented in `LSTMCovariance.py` and `GRUCovariance.py`.

### BEKK-LSTM Kernel Models

`BEKKLSTM` retains time-invariant trainable matrices $C,A,G,B$ and evaluates
the asymmetric BEKK kernel

$$
K_t
=CC^{\top}
+A^{\top}\varepsilon_{t-1}\varepsilon_{t-1}^{\top}A
+G^{\top}\Sigma_{t-1}G
+B^{\top}\eta_{t-1}\eta_{t-1}^{\top}B.
$$

A custom LSTM-style state $c_t$ then modulates $K_t$ in one of three ways.

**Scalar modulation**

$$
m_t=1+\beta\tanh\!\left(w_o^{\top}\tanh(c_t)+b_o\right),
\qquad
\Sigma_t=m_tK_t.
$$

**Vector modulation**

$$
m_t=\mathbf{1}+\beta\tanh\!\left(W_o\tanh(c_t)+b_o\right),
\qquad
\Sigma_t=\mathrm{diag}(m_t)K_t\mathrm{diag}(m_t).
$$

**Convex-mixture modulation**

$$
z_t=[\tanh(c_t);\mathrm{LN}(\mathrm{vech}(K_t))],
\qquad
\lambda_t=\sigma(w_\lambda^{\top}z_t+b_\lambda)\in(0,1),
$$

$$
\Sigma_t^{\mathrm{nn}}=L_t^{\mathrm{nn}}(L_t^{\mathrm{nn}})^{\top},
\qquad
\Sigma_t=(1-\lambda_t)K_t+\lambda_t\Sigma_t^{\mathrm{nn}}.
$$

$L_t^{\mathrm{nn}}$ is produced by a Cholesky head after aligning the neural
branch with the BEKK scale. The gate is a continuous mixture weight, not a
discrete regime-switching state. The implementation is in `bekk_kernel.py`.

### Neural-BEKK with Time-Varying Parameters

`NeuralBekk` uses a single-layer GRU and a parameter head to turn the input
history into admissible BEKK coefficients:

$$
h_{t-1}=\mathrm{GRU}_{\theta}(x_{t-s},\ldots,x_{t-1}),
\qquad
u_t=\mathrm{Head}_{\theta}(h_{t-1}),
\qquad
\gamma_t=\mathcal{C}(u_t).
$$

The two implemented parameterizations are

$$
\gamma_t^{\mathrm{diag}}
=\left(\mathrm{vech}(C_t),a_t,g_t,b_t\right),
\quad
A_t=\mathrm{diag}(a_t),\;
G_t=\mathrm{diag}(g_t),\;
B_t=\mathrm{diag}(b_t),
$$

and

$$
\gamma_t^{\mathrm{full}}
=\left(\mathrm{vech}(C_t),\mathrm{vec}(A_t),
\mathrm{vec}(G_t),\mathrm{vec}(B_t)\right).
$$

Both enter the same time-varying asymmetric recursion:

$$
\Sigma_t
=C_tC_t^{\top}
+A_t^{\top}\varepsilon_{t-1}\varepsilon_{t-1}^{\top}A_t
+G_t^{\top}\Sigma_{t-1}G_t
+B_t^{\top}\eta_{t-1}\eta_{t-1}^{\top}B_t.
$$

The constraint map $\mathcal{C}$ allocates a bounded persistence budget instead
of using unconstrained network outputs directly. In the diagonal model,

$$
\tau_{i,t}=\tau_{\max}\sigma(\widetilde{\tau}_{i,t}),
\qquad
\omega_{i,t}=\mathrm{softmax}(\ell_{i,t}),
$$

$$
(a_{i,t},g_{i,t},b_{i,t})
=\sqrt{\tau_{i,t}\,\omega_{i,t}},
\qquad
a_{i,t}^2+g_{i,t}^2+b_{i,t}^2=\tau_{i,t}<\tau_{\max}<1,
$$

where the square root is component-wise. In the full model, raw matrix
directions are normalized by their spectral norms and scaled so that

$$
\lVert A_t\rVert_2^2+\lVert G_t\rVert_2^2+\lVert B_t\rVert_2^2
\leq\tau_t<\tau_{\max}<1.
$$

This is a conservative sufficient stability restriction that avoids evaluating
Kronecker products during training. The implementation is in `neural_bekk.py`.

## Data

The main sample contains 6,427 common daily observations from August 2000 to
1 May 2026, based on the Yahoo Finance snapshot dated 3 May 2026:

- S&P 500 index (`^GSPC`)
- Gold futures (`GC=F`)
- WTI crude-oil futures (`CL=F`)
- 10-year U.S. Treasury yield (`^TNX`)

The CSV files store decimal returns. For the three price series these are log
returns; the thesis reports them in percent as
$r_{i,t}^{\mathrm{pct}}=100\log(P_{i,t}/P_{i,t-1})$. Because `^TNX` is a yield rather
than a tradable bond price, the repository converts it to a synthetic 10-year
par-bond holding return using daily carry and a second-order
duration-convexity approximation:

$$
\widetilde r_{\mathrm{TNX},t}
\approx
\frac{\widetilde y_{t-1}}{252}
-D_{\mathrm{mod},t-1}\Delta\widetilde y_t
+\frac{1}{2}C_{t-1}(\Delta\widetilde y_t)^2,
\qquad
r_{\mathrm{TNX},t}^{\mathrm{pct}}=100\widetilde r_{\mathrm{TNX},t}.
$$

Recurrent features are standardized from the decimal series. The hybrid BEKK
recursions use percent-scaled residuals internally (`bekk_scale=100`) and map
their covariance outputs back before evaluation. The chronological split is
70% training, 15% validation, and 15% test. Scaling moments are estimated on
the training sample only. After the 40-observation lookback, all models are
aligned to the same final 925 test dates. The test block is never used for
estimation or model selection.

## Estimation and Evaluation

The econometric benchmarks are estimated by Gaussian quasi-maximum likelihood
on the combined training and validation sample. Neural and hybrid models are
trained on the training sample with AdamW, a cosine learning-rate schedule,
gradient clipping, and early stopping on validation negative log-likelihood.
The best validation checkpoint is restored. The principal neural settings are:

- lookback $s=40$ and hidden dimension $h=64$;
- at most 500 epochs;
- early stopping after 50 epochs without an improvement of at least $10^{-4}$;
- 11 training seeds: `42, 0, 1, ..., 9`;
- Gaussian objectives for the normal LSTM/GRU variants;
- covariance-parameterized multivariate Student-t objectives for the remaining
  neural and hybrid variants, with learned $2.01<\nu<100$.

Parameters are frozen over the common out-of-sample block. Each model still
updates its covariance or recurrent state from realized information, but no
model is re-estimated during the test period.

All risk results use the fixed equal-weight portfolio

$$
r_{p,t}=w^{\top}r_t,
\qquad
\sigma_{p,t}^2=w^{\top}\Sigma_tw,
\qquad
w_i=\frac{1}{N}.
$$

The evaluation covers:

- Filtered Historical Simulation VaR and ES with window $W=1000$ at
  $\alpha=0.01$ (and $\alpha=0.05$ as a robustness check);
- Kupiec unconditional-coverage and Christoffersen independence/conditional-
  coverage tests;
- dynamic-quantile and Expected Shortfall backtests;
- the strictly consistent joint Fissler-Ziegel VaR-ES loss;
- Harvey-Leybourne-Newbold-modified Diebold-Mariano comparisons;
- 90% model confidence sets based on stationary bootstrap resampling;
- portfolio and matrix QLIKE robustness comparisons.

## Main Findings

Under the main fixed-parameter design, the three lowest mean out-of-sample FZ0
losses are obtained by the diagonal asymmetric Neural-BEKK, the convex-mixture
BEKK-LSTM, and the scalar BEKK-LSTM. The leading hybrids also improve dynamic
VaR calibration relative to the classical benchmarks.

These differences are not decisive. The 90% model confidence set retains almost
all specifications, some leading hybrids fail two-sided tests of exact ES
calibration, and rankings vary across loss functions and alternative sample
designs. Within the Neural-BEKK family, the diagonal restriction outperforms the
full time-varying parameterization on mean FZ0 and QLIKE loss; the additional
off-diagonal flexibility provides no observed out-of-sample benefit in the main
sample.

The convex-mixture gate is BEKK-dominant on average and shifts further toward
the structured kernel during the major stress episodes. This is descriptive
evidence about learned allocation, not a causal attribution of forecast gains.

## Repository Structure

```text
.
|-- neural_bekk.py              # Diagonal/full time-varying Neural-BEKK
|-- bekk_kernel.py              # BEKK-LSTM cell and modulation heads
|-- LSTMCovariance.py           # Direct LSTM covariance benchmark
|-- GRUCovariance.py            # Direct GRU covariance benchmark
|-- training_functions.py       # Likelihoods, training, early stopping, checkpoints
|-- backtesting.py              # VaR/ES tests, FZ loss, and DM comparisons
|-- risk_metrics.py             # FHS and parametric VaR/ES construction
|-- return_transforms.py        # Treasury yield-to-return transformation
|-- run_models.ipynb            # Main empirical pipeline
|-- run_models_code_only.py     # Exported notebook code
|-- evaluate_runs.py            # Cross-seed aggregation and MCS evaluation
|-- r/
|   |-- bekks.R                 # Symmetric/asymmetric BEKK estimation
|   |-- dcc.R                   # DCC/ADCC estimation
|   `-- output/                 # Saved econometric covariance forecasts
|-- scripts/                    # Training, robustness, tables, and figures
|-- data/                       # Versioned raw data and chronological splits
|-- results/runs/               # Per-snapshot, per-seed checkpoints and results
|-- figures/                    # Thesis figures
|-- tables/                     # Thesis tables
`-- tests/                      # Model, transformation, and backtesting tests
```

## Setup

The submitted Python environment is pinned in `requirements.txt`. From the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The econometric benchmarks and R-backed backtests additionally require:

```r
install.packages(c("BEKKs", "rugarch", "rmgarch", "esback", "segMGarch", "MCS"))
```

`rpy2` is required when Python invokes the R-based dynamic-quantile and ES
backtests.

## Reproducing the Analysis

For the ordered artifact-level and full-training workflows, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Run the test suite first:

```bash
python -m unittest discover -s tests -v
```

Execute the main notebook for the default seed (`MODEL_SEED=42`):

```bash
MODEL_SEED=42 jupyter nbconvert \
  --to notebook \
  --execute run_models.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --stdout > /dev/null
```

Run all 11 neural-model seeds:

```bash
bash scripts/run_all_seeds.sh
```

Train only the full asymmetric Neural-BEKK follow-up:

```bash
python -u scripts/train_neural_bekk_full.py
```

Generate representative figures and aggregate results:

```bash
python scripts/make_section3_figures.py
python scripts/make_section4_loss_figures.py
python evaluate_runs.py --no-overleaf
```

The pipeline uses `DATA_ID=2026-05-03` for the main snapshot and writes each run
to

```text
results/runs/asof_<DATA_ID>_seed<MODEL_SEED>/
```

Saved checkpoints, backtest summaries, covariance forecasts, figures, and
tables are already included, so the reported artifacts can be inspected without
retraining.

> **Path portability:** the notebook export and `scripts/run_all_seeds.sh`
> retain absolute paths from the thesis environment. If the repository is
> cloned elsewhere, update `project_dir` in `run_models.ipynb` (or
> `run_models_code_only.py`) and the `cd` target in `scripts/run_all_seeds.sh`
> before running the full pipeline.

## Citation

If this repository contributes to academic work, please cite the accompanying
thesis:

```bibtex
@mastersthesis{dinkela2026hybrid,
  author = {Dinkela, Henning Siept},
  title  = {Hybrid Neural-MGARCH Models for Volatility and Financial Risk Forecasting},
  school = {Georg-August University of Göttingen},
  year   = {2026}
}
```
