#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from backtesting import build_fz_loss_matrix, run_backtest_suite, run_fz_dm_comparison_plan

from gen_seq_data import SequenceDataset
from vech import vech, unvech

from torch.utils.data import Dataset, DataLoader
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import time
from datetime import datetime
from zoneinfo import ZoneInfo
import json

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ## Seeding and Versionierung

# In[823]:


# SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

SEED = int(os.environ.get("MODEL_SEED", "42"))

DATA_AS_OF = "2026-05-03"
DATA_ID = DATA_AS_OF

AS_OF_DATE = pd.Timestamp(DATA_AS_OF)
YFINANCE_END = (AS_OF_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

RUN_ID = f"asof_{DATA_AS_OF}_seed{SEED}"
RUN_TIMESTAMP = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(SEED)

project_dir = Path("/Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk")
data_dir = project_dir / "data"
output_dir = project_dir / "r/output"
run_dir = project_dir / "results" / "runs" / RUN_ID
models_dir = run_dir / "models"

data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)
run_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print("DATA_ID:", DATA_ID)
print("RUN_ID:", RUN_ID)
print("YFINANCE_END:", YFINANCE_END)
print("RUN_TIMESTAMP:", RUN_TIMESTAMP)
print("device:", device)


# In[824]:


from training_functions import save_model_checkpoint

RUN_METADATA = {
    "seed": SEED,
    "data_id": DATA_ID,
    "data_as_of": DATA_AS_OF,
    "run_id": RUN_ID,
    "run_timestamp": RUN_TIMESTAMP,
    "device": str(device),
}


# ## Price Data

# In[825]:


macro_tickers = [
    "^GSPC",  # Equity (S&P 500)
    "GC=F",   # Gold
    "CL=F",   # Oil
    "^TNX"    # 10Y yield
]

macro_btc_tickers = [
    "^GSPC",
    "GC=F",
    "CL=F",
    "^TNX",
    "BTC-USD"
]


# In[826]:


macro_data = yf.download(
    macro_tickers,
    start="1900-01-01",
    end=YFINANCE_END,          # yfinance end ist exklusiv
    auto_adjust=False,
    progress=False,
)

macro_btc_data = yf.download(
    macro_btc_tickers,
    start="1900-01-01",
    end=YFINANCE_END,
    auto_adjust=False,
    progress=False,
)

macro_data.to_csv(data_dir / f"macro_data_{DATA_ID}.csv")
macro_btc_data.to_csv(data_dir / f"macro_btc_data_{DATA_ID}.csv")

print("Saved:", data_dir / f"macro_data_{DATA_ID}.csv")
print("Saved:", data_dir / f"macro_btc_data_{DATA_ID}.csv")
print("macro_data last date:", macro_data.dropna(how="all").index.max())
print("macro_btc_data last date:", macro_btc_data.dropna(how="all").index.max())


# In[827]:


macro_data


# In[828]:


close = macro_data["Close"].copy().dropna()


# In[829]:


len(close)


# In[830]:


close.plot(title="Close Price", subplots=True, figsize=(12, 6))


# In[831]:


np.log(close).plot(title="Close Price", subplots=True, figsize=(12, 6))


# In[832]:


logret = np.log(close).diff() # log(P_t) - log(P_{t-1})) = log(P_t / P_{t-1})
logret["^TNX"] = close["^TNX"].diff() # r_t(yiel) = P_t(yield) - P_{t-1}(yield)

# Aufräumen: erste Zeile NaN durch diff, ggf. weitere NaNs durch Datenlücken
logret = logret.dropna()

logret.head(), logret.tail()


# In[833]:


len(logret)


# ## Data Export

# In[834]:


import os
from pathlib import Path


# In[835]:


project_dir = Path("/Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk")
data_dir = project_dir / "data"
output_dir = project_dir / "r/output"

data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)


# In[836]:


# Versionierte Datensplits nicht loeschen.
# Dateien werden ueber DATA_ID eindeutig benannt.


# In[837]:


logret


# In[838]:


logret.plot(subplots=True, figsize=(12, 6))


# In[839]:


logret.hist(bins=50, figsize=(12, 6))


# In[840]:


for col in logret.columns:
    plot_acf(logret[col], lags=40, title=f"ACF of log returns for {col}")
    plot_pacf(logret[col], lags=40, title=f"PACF of log returns for {col}")


# In[841]:


# squared returns
for col in logret.columns:
    plot_acf(logret[col]**2, lags=40, title=f"ACF of squared returns for {col}")
    plot_pacf(logret[col]**2, lags=40, title=f"PACF of squared returns for {col}")


# In[842]:


from statsmodels.tsa.stattools import adfuller

for col in logret.columns:
    print(f"ADF test for {col}:")
    result = adfuller(logret[col])
    print(f"ADF statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("---")



# In[843]:


#feature_cols = macro_tickers.split(" ")


# In[844]:


macro_tickers


# In[845]:


R = logret[macro_tickers].values.astype(np.float32)   # (N,d)

# Outer products pro t: (N,d,1) * (N,1,d) -> (N,d,d)
U = R[:, :, None] * R[:, None, :]


# In[846]:


cross_returns = vech(torch.tensor(U))
cross_returns.shape


# In[847]:


logret.shape


# In[848]:


input_data = np.concatenate([logret[macro_tickers].values, cross_returns], axis=1)
input_df = pd.DataFrame(input_data, columns=macro_tickers + [f"cross_{i+1}" for i in range(cross_returns.shape[1])])


# In[849]:


logret.index


# In[850]:


input_df.index = logret.index


# In[851]:


input_df


# In[852]:


cross_df = pd.DataFrame(cross_returns, columns=[f"cross_{i+1}" for i in range(cross_returns.shape[1])])


# In[853]:


cross_df


# In[ ]:


def train_val_test_split(df, train_size=0.7, val_size=0.15):
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


# In[855]:


train_df, val_df, test_df = train_val_test_split(input_df)


# In[856]:


train_df.index.name


# In[857]:


import datetime


# In[858]:


exports = {
    "close": close[macro_tickers],
    "logret": logret[macro_tickers],
    "train_df": train_df[macro_tickers],
    "val_df": val_df[macro_tickers],
    "test_df": test_df[macro_tickers],
}

for name, df_export in exports.items():
    df_export = df_export.copy()
    df_export.index.name = "Date"
    df_export.to_csv(data_dir / f"{name}_{DATA_ID}.csv")

print(f"Exportiert mit DATA_ID: {DATA_ID}")
print(f"RUN_ID fuer Modellergebnisse: {RUN_ID}")


# ## Vola

# Wir modellieren nun
# $\mathbf r_{t+1}\mid\mathcal F_t \sim \mathcal N(\mathbf 0, \Sigma_{t+1}),$
# und lassen das LSTM direkt eine gueltige Kovarianzmatrix $\Sigma_{t+1}$ ausgeben.
# 
# Dazu parametrisieren wir $\Sigma_{t+1} = L_{t+1}L_{t+1}^\top$ mit einer unteren Dreiecksmatrix $L_{t+1}$ (Cholesky-Form).
# Die Diagonale von $L_{t+1}$ wird mit `softplus` strikt positiv gemacht.
# 

# In[859]:


#feature_cols = tickers.split(" ")
mu = train_df.mean()
sigma = train_df.std() # vielleicht nicht durch die Std teilen

def normalize(df, mu, sigma, with_std=True):
    if with_std:
        return (df - mu) / sigma
    else:
        return df - mu


# ## Hyper Paramters

# In[860]:


lookback = 40
batchsize = 64
hidden_size = 64
dropout = 0.1
lr = 3e-4
bekk_lr = 1e-4
bekk_mix_lr = 5e-5
neural_bekk_lr = 1e-4
bekk_jitter = 1e-4
bekk_device = "cpu"
early_stopping_patience = 50
early_stopping_min_delta = 1e-4


# ## Data Loader

# In[861]:


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# In[862]:


train_norm = normalize(train_df, mu, sigma, with_std=True)
val_norm = normalize(val_df, mu, sigma, with_std=True)
test_norm = normalize(test_df, mu, sigma, with_std=True)

vol_train_ds = SequenceDataset(
    train_norm.values,
    train_norm[macro_tickers].values,
    lookback,
)
vol_val_ds = SequenceDataset(
    val_norm.values,
    val_norm[macro_tickers].values,
    lookback,
)
vol_test_ds = SequenceDataset(
    test_norm.values,
    test_norm[macro_tickers].values,
    lookback,
)

vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
vol_val_loader = DataLoader(vol_val_ds, batch_size=batchsize, shuffle=False)
vol_test_loader = DataLoader(vol_test_ds, batch_size=batchsize, shuffle=False)


# In[863]:


len(vol_train_ds), len(vol_val_ds), len(vol_test_ds)


# ```train_cov```ggf. zur Initialisierung nutzen. 

# In[864]:


train_cov = torch.cov(torch.tensor(train_norm[macro_tickers].values, dtype=torch.float32,).T)  # sample / long-term covariance from training data
train_cov


# In[865]:


from LSTMCovariance import LSTMCovariance
from training_functions import train_covariance_model, gaussian_nll, student_nll


# In[866]:


seed_everything(SEED)

k = len(macro_tickers)*(len(macro_tickers)+1)//2 # Anzahl an Cross Returns, die zusätzlich zu den normalen Returns als Input-Features für die Kovarianzmodellierung genutzt werden

cov_model = LSTMCovariance(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=hidden_size, num_layers=2, dropout=dropout)
cov_model, hist = train_covariance_model(
    cov_model,
    vol_train_loader,
    vol_val_loader,
    epochs=500,
    lr=lr,
    plateau_patience=20,
    device=device,
    scheduler_type="cosine",  # oder "cosine"
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
)

save_model_checkpoint(
    model_name="lstm_normal",
    models_dir=models_dir,
    model=cov_model,
    history=hist,
    run_metadata=RUN_METADATA,
    loss_fn=None,
    config={
        "model_class": "LSTMCovariance",
        "distribution": "gaussian",
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": hidden_size,
        "num_layers": 2,
        "dropout": dropout,
        "lookback": lookback,
        "batchsize": batchsize,
        "epochs": 500,
        "lr": lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    },
)


# In[867]:


from plots import plot_loss


# In[868]:


plot_loss(hist)


# In[869]:


sigma


# In[870]:


def predict_sigma(model, loader, device='cpu'):
    model.eval()
    sigmas = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            sigma_pred, _ = model(xb)
            sigmas.append(sigma_pred.cpu().numpy())
            targets.append(yb.numpy())

    return np.concatenate(sigmas, axis=0), np.concatenate(targets, axis=0)


sigma_val_scaled, y_val_scaled = predict_sigma(cov_model, vol_val_loader, device=device)
sigma_test_scaled, y_test_scaled = predict_sigma(cov_model, vol_test_loader, device=device)

# Transform covariance back to original return scale:
# if z = (r - mu)/std, then Cov(r) = D Cov(z) D with D = diag(std)

def scale_back(sigma_scaled, sigma=None, feature_cols=None):
    if sigma is not None and feature_cols is not None:
        std_vec = sigma[feature_cols].values.astype(np.float32)
        return sigma_scaled * std_vec[None, :, None] * std_vec[None, None, :]
    else:
        return sigma_scaled

sigma_val_real = scale_back(sigma_val_scaled, sigma=sigma, feature_cols=macro_tickers)
sigma_test_real = scale_back(sigma_test_scaled, sigma=sigma, feature_cols=macro_tickers)

print('sigma_val_scaled:', sigma_val_scaled.shape)
print('sigma_test_scaled:', sigma_test_scaled.shape)
print('sigma_test_real:', sigma_test_real.shape)
print('sigma_val_real:', sigma_val_real.shape)


# In[871]:


vol_train_eval_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=False) # ohne Shuffle, für Plots


# In[872]:


sigma_train_scaled, y_train_scaled = predict_sigma(cov_model, vol_train_eval_loader, device=device)
sigma_train_real = scale_back(sigma_train_scaled, sigma=sigma, feature_cols=macro_tickers)


# In[873]:


# Quick diagnostic plot: predicted conditional variances
# Build a time index + diagonal variance frame first
test_index_vol = test_df.index[lookback:]
sigma_test_diag_df = pd.DataFrame(
    {
        f'Var({macro_tickers[i]})': sigma_test_real[:, i, i]
        for i in range(len(macro_tickers))
    },
    index=test_index_vol,
)

fig, axes = plt.subplots(len(macro_tickers), 1, figsize=(12, 6), sharex=True)
if len(macro_tickers) == 1:
    axes = [axes]

for i, col in enumerate(macro_tickers):
    axes[i].plot(sigma_test_diag_df.index, sigma_test_diag_df[f'Var({col})'])
    axes[i].set_title(f'Predicted conditional variance: {col}')

plt.tight_layout()
plt.show()


# In[874]:


from statistics import NormalDist
from scipy.stats import t
from plots import plot_var


# In[875]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split", df=train_df, pred_vol=sigma_train_real)


# In[876]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Validation-Split", df=val_df, pred_vol=sigma_val_real)


# In[877]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Test-Split", df=test_df, pred_vol=sigma_test_real)


# In[878]:


train_df.shape, val_df.shape, test_df.shape


# # Portfolio Berechnungen

# Portfolio Weights & Portfolio Return $r_{p,t} = w^\top \cdot r_t$
# 
# $w = (w_1, w_2, \dots, w_n)^\top$ mit $\sum_i^n w_i = 1$
# 
# Portfolio Variance:
# 
# $\sigma^2_{p,t} = w^\top \Sigma_t w$
# 

# In[879]:


w = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32).to(device)  # gleichgewichtetes Portfolio


# In[880]:


#w = torch.tensor([0.5, 0.4, 0.1])


# In[881]:


w.shape


# In[882]:


logret.shape


# In[883]:


torch.tensor(logret.values).shape


# In[884]:


logret_tensor = torch.tensor(logret.values, dtype=torch.float32, device=w.device)


# In[885]:


w.shape


# In[886]:


logret_tensor.shape


# In[887]:


portfolio_logrets = (w * logret_tensor).sum(dim=1)


# In[888]:


portfolio_logrets.shape


# In[889]:


portfolio_logrets_np = portfolio_logrets.detach().cpu().numpy()
plt.figure(figsize=(12, 4))
plt.plot(logret.index, portfolio_logrets_np, label="Portfolio Log-Returns")
plt.title("Portfolio Log-Returns (equal weights)")
plt.legend()
plt.tight_layout()
plt.show()


# In[890]:


#w = np.array([0.5, 0.4, 0.1], dtype=np.float32)
#w


# In[891]:


def calc_portfolio_variance(df_split, sigma_split, cols, lookback, w):
    idx = df_split.index[lookback:]
    r = df_split.loc[idx, cols].values  # (N-lookback, d)
    r_p = (w*r).sum(axis=1)  # (N-lookback,) Portfolio-Logreturns
    tmp = sigma_split @ w          # (T, d)  = Σ_t w
    var_p = (w * tmp).sum(axis=1) # (1,) = w^T (Σ_t w) Portfolio-Varianz pro t
    vol_p = np.sqrt(var_p)

    return idx, r_p, var_p, vol_p


# ## LSTM mit Normal Loss

# In[892]:


w_np = w.detach().cpu().numpy()


# In[893]:


sigma_train_real.shape


# In[894]:


idx_test, rp_test, varp_test, volp_test = calc_portfolio_variance(
    test_df, sigma_test_real, macro_tickers, lookback, w_np
)

idx_val, rp_val, varp_val, volp_val = calc_portfolio_variance(
    val_df, sigma_val_real, macro_tickers, lookback, w_np
)

idx_train, rp_train, varp_train, volp_train = calc_portfolio_variance(
    train_df, sigma_train_real, macro_tickers, lookback, w_np
)


# In[895]:


r = np.asarray(rp_test, dtype=float)
v = np.asarray(volp_test, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[896]:


def make_portfolio_df(idx, rp, varp, volp):
    return pd.DataFrame(
        {
            "r_p": np.asarray(rp, dtype=float),
            "var_p": np.asarray(varp, dtype=float),
            "vol_p": np.asarray(volp, dtype=float),
        },
        index=idx,
    )


# In[897]:


portfolio_test = make_portfolio_df(idx_test, rp_test, varp_test, volp_test)
portfolio_val = make_portfolio_df(idx_val, rp_val, varp_val, volp_val)
portfolio_train = make_portfolio_df(idx_train, rp_train, varp_train, volp_train)


# In[898]:


test_df.columns


# In[899]:


macro_tickers


# In[900]:


plot_var(alpha=0.01, lookback=lookback, view="Test-Split", portfolio=True, portfolio_df=portfolio_test)


# In[901]:


plot_var(alpha=0.01, lookback=lookback, view="Val-Split", portfolio=True, portfolio_df=portfolio_val)


# In[902]:


plot_var(alpha=0.01, lookback=lookback, view="Train-Split", portfolio=True, portfolio_df=portfolio_train)


# ## LSTM mit Student-t Loss

# ### Training

# Modell mit Studen-t-Verteilung

# $$r_t \sim t_\nu(0, S_t)$$
# 
# $$ S_t = \frac{\nu}{\nu-2} \Sigma_t = \frac{\nu}{\nu-2} L_t L_t^\top = (\sqrt{\frac{\nu}{\nu-2}} L) \cdot (\sqrt{\frac{\nu}{\nu-2}} L)^\top$$

# In[903]:


from training_functions import StudentTLoss


# In[904]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[905]:


k = len(macro_tickers)*(len(macro_tickers)+1)//2 # Anzahl an Cross Returns, die zusätzlich zu den normalen Returns als Input-Features für die Kovarianzmodellierung genutzt werden

cov_model_t = LSTMCovariance(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=hidden_size, num_layers=2, dropout=dropout)
cov_model_t, hist_t = train_covariance_model(
    cov_model_t,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss,
    loss_kwargs=None,
    epochs=500,
    lr=lr,
    plateau_patience=20,
    device=device,
    scheduler_type="cosine",  # oder "cosine"
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
)

save_model_checkpoint(
    model_name="lstm_student",
    models_dir=models_dir,
    model=cov_model_t,
    history=hist_t,
    run_metadata=RUN_METADATA,
    loss_fn=student_loss,
    config={
        "model_class": "LSTMCovariance",
        "distribution": "student_t",
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": hidden_size,
        "num_layers": 2,
        "dropout": dropout,
        "lookback": lookback,
        "batchsize": batchsize,
        "epochs": 500,
        "lr": lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "nu": float(student_loss.nu.detach().cpu()),
    },
)


# In[906]:


learned_nu = float(student_loss.nu.detach().cpu())
student_kwargs = {"nu": learned_nu}


# In[907]:


plot_loss(hist_t)


# In[908]:


sigma_val_scaled_t, y_val_scale_t = predict_sigma(cov_model_t, vol_val_loader, device=device)
sigma_test_scaled_t, y_test_scaled_t = predict_sigma(cov_model_t, vol_test_loader, device=device)

sigma_val_real_t = scale_back(sigma_val_scaled_t, sigma=sigma, feature_cols=macro_tickers)
sigma_test_real_t = scale_back(sigma_test_scaled_t, sigma=sigma, feature_cols=macro_tickers)


# In[909]:


vol_train_eval_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=False)
sigma_train_scaled_t, y_train_scaled_t = predict_sigma(cov_model_t, vol_train_eval_loader, device=device)
sigma_train_real_t = scale_back(sigma_train_scaled_t, sigma=sigma, feature_cols=macro_tickers)


# In[910]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL)", df=train_df, pred_vol=sigma_train_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[911]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL)", df=val_df, pred_vol=sigma_val_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[912]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL)", df=test_df, pred_vol=sigma_test_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# ### Portfolio Returns

# - Hier müssten jetzt noch mal die Portfolio Returns & Volas mit ```make_portfolio_df``` berechnet werden für Student-t Verteilung!

# In[913]:


idx_test_t, rp_test_t, varp_test_t, volp_test_t = calc_portfolio_variance(
    test_df, sigma_test_real_t, macro_tickers, lookback, w_np
)

idx_val_t, rp_val_t, varp_val_t, volp_val_t = calc_portfolio_variance(
    val_df, sigma_val_real_t, macro_tickers, lookback, w_np
)

idx_train_t, rp_train_t, varp_train_t, volp_train_t = calc_portfolio_variance(
    train_df, sigma_train_real_t, macro_tickers, lookback, w_np
)


# In[914]:


r = np.asarray(rp_test_t, dtype=float)
v = np.asarray(volp_test_t, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[915]:


portfolio_test_t = make_portfolio_df(idx_test_t, rp_test_t, varp_test_t, volp_test_t)
portfolio_val_t = make_portfolio_df(idx_val_t, rp_val_t, varp_val_t, volp_val_t)
portfolio_train_t = make_portfolio_df(idx_train_t, rp_train_t, varp_train_t, volp_train_t)


# Modell mit skewed Studen-t-Verteilung

# ## Filtered Historical Simulation

# - bisher FHS nur mit Normal Volas

# In[916]:


from risk_metrics import fhs_var_es


# In[917]:


alpha = 0.01
window = 1000  # z.B. 250/500/1000 testen


# In[918]:


len(train_df), len(val_df), len(test_df)


# ### Normal Loss (Portfolio)

# In[919]:


z_train = portfolio_train["r_p"].values / np.clip(portfolio_train["vol_p"].values, 1e-12, None) # clip sichert nur nach unten ab (0 division)
z_val = portfolio_val["r_p"].values / np.clip(portfolio_val["vol_p"].values, 1e-12, None)
z_test = portfolio_test["r_p"].values / np.clip(portfolio_test["vol_p"].values, 1e-12, None)
z_train_val =np.concatenate([z_train, z_val])


var_fhs_test, es_fhs_test, hits_test = fhs_var_es(
    z_hist_init=z_train_val, # nimmt die letzen #window Werte als Historie
    r_oos=portfolio_test["r_p"].values,
    vol_oos=portfolio_test["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_test = hits_test.mean()
print(f"FHS Test hit rate = {hit_rate_test:.3%} (target {alpha:.3%})")


# In[920]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_test.index, portfolio_test["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_test.index, var_fhs_test, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_test.index, var_fhs_test, np.zeros_like(var_fhs_test), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Test)")
plt.legend()
plt.tight_layout()
plt.show()


# - Falls FHS auf train genutzt werden soll, muss zuvor ein z_train_burn_in genutzt werden wo zb die ersten 500 z_t aus train genutzt werden für die historie und von dort aus wird dann der VaR rollierend berechnet!

# In[921]:


z_train_burn_in = z_train[:window]


# In[922]:


var_fhs_train, es_fhs_train, hit_train = fhs_var_es(
    z_hist_init=z_train_burn_in, # custom burn-in für Historie nötig
    r_oos=portfolio_train["r_p"][window:].values,
    vol_oos=portfolio_train["vol_p"][window:].values,
    alpha=alpha,
    window=window
)

hit_rate_train = hit_train.mean()
print(f"FHS Train hit rate = {hit_rate_train:.3%} (target {alpha:.3%})")


# In[923]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_train.index, portfolio_train["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_train[window:].index, var_fhs_train, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_train[window:].index, var_fhs_train, np.zeros_like(var_fhs_train), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Train)")
plt.legend()
plt.tight_layout()
plt.show()


# In[924]:


var_fhs_val, es_fhs_val, hit_val = fhs_var_es(
    z_hist_init=z_train,
    r_oos=portfolio_val["r_p"].values,
    vol_oos=portfolio_val["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_val = hit_val.mean()
print(f"FHS Val hit rate = {hit_rate_val:.3%} (target {alpha:.3%})")


# In[925]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_val.index, portfolio_val["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_val.index, var_fhs_val, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_val.index, var_fhs_val, np.zeros_like(var_fhs_val), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Validation)")
plt.legend()
plt.tight_layout()
plt.show()


# ### Student-t Loss (Portfolio)

# In[926]:


z_train_t = portfolio_train_t["r_p"].values / np.clip(portfolio_train_t["vol_p"].values, 1e-12, None) # clip sichert nur nach unten ab (0 division)
z_val_t = portfolio_val_t["r_p"].values / np.clip(portfolio_val_t["vol_p"].values, 1e-12, None)
z_test_t = portfolio_test_t["r_p"].values / np.clip(portfolio_test_t["vol_p"].values, 1e-12, None)
z_train_val_t =np.concatenate([z_train_t, z_val_t])


var_fhs_test_t, es_fhs_test_t, hits_test_t = fhs_var_es(
    z_hist_init=z_train_val_t, # nimmt die letzen #window Werte als Historie
    r_oos=portfolio_test_t["r_p"].values,
    vol_oos=portfolio_test_t["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_test_t = hits_test_t.mean()
print(f"FHS Test hit rate = {hit_rate_test_t:.3%} (target {alpha:.3%})")


# # MGARCH-LSTM mit BEKK Kernel Function (Zhao et al. 2024)

# $$
# \textbf{Original LSTM:}
# $$
# 
# $$
# f_t = \sigma_g\!\left(W_f x_t 
#     + U_f h_{t-1} 
#     + b_f\right), \\
# 
# i_t = \sigma_g\!\left(W_i x_t 
#     + U_i h_{t-1} 
#     + b_i\right), \\
# 
# o_t = \sigma_g\!\left(W_o x_t 
#     + U_o h_{t-1} 
#     + b_o\right), \\
# 
# \tilde{c}_t = \sigma_c\!\left(W_c x_t 
#     + U_c h_{t-1} 
#     + b_c\right), \\
# 
# c_t = f_t \odot c_{t-1} 
#     + i_t \odot \tilde{c}_t, \\
# 
# h_t = o_t \odot \tanh(c_t)
# $$

# $s_{t-1} := \operatorname{vech}(\Sigma_{t-1}) $
# 
# $x_{t} := [
#   \text{standardisierte Returns,
#   standardisierte Cross-Return-Features,
#   ggf. weitere standardisierte Features}
# ]$

# $$
# \textbf{LSTM mit BEKK Kernel:}
# $$
# 
# $$
# f_t = \sigma_g\!\left(W_f x_{t} 
#       + U_f s_{t-1} 
#       + b_f\right), \\
# i_t = \sigma_g\!\left(W_i x_{t} 
#       + U_i s_{t-1} 
#       + b_i\right), \\
# \tilde c_t = \sigma_c\!\left(W_c x_{t} 
#       + U_c s_{t-1} 
#       + b_c\right), \\
# c_t = f_t \odot c_{t-1} 
#       + i_t \odot \tilde c_t.
# $$
# 
# The output gate can then be reinterpreted as a BEKK kernel output
# $$
# o_t := \mathcal{K}_{\text{BEKK}}(\varepsilon_{t-1}, \Sigma_{t-1}; \Theta), \\
# \mathcal{K}_{\text{BEKK}} = C C^\top 
#       + A^\top \varepsilon_{t-1}\varepsilon_{t-1}^\top A 
#       + B^\top \Sigma_{t-1} B.
# $$

# $$
# \textbf{Scalar modulation:}
# $$
# 
# $$m_t = 1 + \beta \tanh(w_o^\top \tanh(c_t) + b_o)$$
# 
# $$\Sigma_t
# = m_t \times o_t = 
# \underbrace{m_t}_{\text{LSTM}}
# \times
# \underbrace{
# \left(
# CC^\top
# +
# A^\top \varepsilon_{t-1}\varepsilon_{t-1}^\top A
# +
# B^\top \Sigma_{t-1} B
# \right)
# }_{\text{BEKK(1,1)}}$$

# $$
# \textbf{Vector modulation:}
# $$
# 
# $$
# m_t = \mathbf{1}_d + \beta \tanh\big(W_o \tanh(c_t) + b_o\big), \quad m_t \in (1-\beta, 1+\beta)^d
# $$
# 
# $$
# \Sigma_t = \operatorname{diag}(m_t)\, o_t\, \operatorname{diag}(m_t)
# $$

# $$
# \textbf{Convex mixture (regime-switching) modulation:}
# $$
# 
# $$
# \alpha_t = \sigma\left(
# W_\alpha 
# \begin{bmatrix}
# \tanh(c_t) \\
# \operatorname{vech}(o_t)
# \end{bmatrix}
# + b_\alpha
# \right), 
# \quad \alpha_t \in (0,1)
# $$
# 
# $$
# \Sigma_t^{\text{LSTM}} = L_t L_t^\top
# $$
# 
# $$
# \Sigma_t = (1-\alpha_t)\, o_t + \alpha_t\, \Sigma_t^{\text{LSTM}}
# $$

# In[927]:


from bekk_kernel import BEKKCell, BEKKLSTM
from neural_bekk import NeuralBekk


# - Es müssen neue Dataset & Dataloader angelegt werden, da die MGARCH Kernel Function ein BEKK(1,1) ist. Daher ist der bisherige ```lookback=60``` nicht passend.

# In[928]:


#loss_kwargs = {"nu": 10.0}


# ```train_cov```ggf. zur Initialisierung nutzen. 

# In[929]:


train_cov = torch.cov(torch.tensor(train_norm[macro_tickers].values, dtype=torch.float32,).T)  # sample / long-term covariance from training data
train_cov


# In[930]:


Sigma0 = torch.tensor(np.cov((train_df[macro_tickers].values * 100).T),
                      dtype=torch.float32)


# In[931]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_bekk = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# ## Scalar Modulation

# In[932]:


bekk_kernel_lstm = BEKKLSTM(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=hidden_size, asym=True, Sigma0=train_cov, modulation="scalar", return_std=sigma[macro_tickers], bekk_scale=100.0, sigma0_in_bekk_scale=False, gate_cov_layernorm=True, init_c_from_sigma0=True, jitter=bekk_jitter)
bekk_kernel_lstm, hist_bekk = train_covariance_model(
    bekk_kernel_lstm,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss_bekk,
    loss_kwargs=None,
    epochs=500,
    lr=bekk_lr,
    plateau_patience=20,
    device=bekk_device,
    scheduler_type="cosine",  # oder "cosine"
    grad_clip_max_norm=0.5,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
)

save_model_checkpoint(
    model_name="bekk_lstm_scalar",
    models_dir=models_dir,
    model=bekk_kernel_lstm,
    history=hist_bekk,
    run_metadata=RUN_METADATA,
    loss_fn=student_loss_bekk,
    config={
        "model_class": "BEKKLSTM",
        "distribution": "student_t",
        "modulation": "scalar",
        "asym": True,
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": hidden_size,
        "lookback": lookback,
        "batchsize": batchsize,
        "epochs": 500,
        "lr": bekk_lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "device": bekk_device,
        "bekk_scale": 100.0,
        "jitter": bekk_jitter,
        "sigma0_in_bekk_scale": False,
        "gate_cov_layernorm": True,
        "init_c_from_sigma0": True,
        "nu": float(student_loss_bekk.nu.detach().cpu()),
    },
)


# In[933]:


learned_nu_bekk = float(student_loss_bekk.nu.detach().cpu())
bekk_student_kwargs = {"nu": learned_nu_bekk}


# In[934]:


learned_nu, learned_nu_bekk


# In[935]:


plot_loss(hist_bekk)


# In[936]:


sigma_val_scaled_bekk, y_val_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_val_loader, device=bekk_device)
sigma_val_real_bekk = scale_back(sigma_val_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk, y_test_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_test_loader, device=bekk_device)
sigma_test_real_bekk = scale_back(sigma_test_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk, y_train_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_train_eval_loader, device=bekk_device)
sigma_train_real_bekk = scale_back(sigma_train_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)


# In[937]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel)", df=train_df, pred_vol=sigma_train_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[938]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel)", df=val_df, pred_vol=sigma_val_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[939]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel)", df=test_df, pred_vol=sigma_test_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# ### Portfolio

# In[940]:


idx_test_bekk, rp_test_bekk, varp_test_bekk, volp_test_bekk = calc_portfolio_variance(
    test_df, sigma_test_real_bekk, macro_tickers, lookback, w_np
)

idx_val_bekk, rp_val_bekk, varp_val_bekk, volp_val_bekk = calc_portfolio_variance(
    val_df, sigma_val_real_bekk, macro_tickers, lookback, w_np
)

idx_train_bekk, rp_train_bekk, varp_train_bekk, volp_train_bekk = calc_portfolio_variance(
    train_df, sigma_train_real_bekk, macro_tickers, lookback, w_np
)


# In[941]:


r = np.asarray(rp_test_bekk, dtype=float)
v = np.asarray(volp_test_bekk, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[942]:


portfolio_test_bekk = make_portfolio_df(idx_test_bekk, rp_test_bekk, varp_test_bekk, volp_test_bekk)
portfolio_val_bekk = make_portfolio_df(idx_val_bekk, rp_val_bekk, varp_val_bekk, volp_val_bekk)
portfolio_train_bekk = make_portfolio_df(idx_train_bekk, rp_train_bekk, varp_train_bekk, volp_train_bekk)


# ### FHS

# In[943]:


z_train_bekk = portfolio_train_bekk["r_p"].values / np.clip(portfolio_train_bekk["vol_p"].values, 1e-12, None) # clip sichert nur nach unten ab (0 division)
z_val_bekk = portfolio_val_bekk["r_p"].values / np.clip(portfolio_val_bekk["vol_p"].values, 1e-12, None)
z_test_bekk = portfolio_test_bekk["r_p"].values / np.clip(portfolio_test_bekk["vol_p"].values, 1e-12, None)
z_train_val_bekk =np.concatenate([z_train_bekk, z_val_bekk])


var_fhs_test_bekk, es_fhs_test_bekk, hits_test_bekk = fhs_var_es(
    z_hist_init=z_train_val_bekk, # nimmt die letzen #window Werte als Historie
    r_oos=portfolio_test_bekk["r_p"].values,
    vol_oos=portfolio_test_bekk["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_test_bekk = hits_test_bekk.mean()
print(f"FHS Test hit rate = {hit_rate_test_bekk:.3%} (target {alpha:.3%})")


# In[944]:


len(close)


# ## Vector Modulation

# In[945]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_bekk_vec = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[946]:


bekk_kernel_lstm_vec = BEKKLSTM(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=hidden_size, asym=True, Sigma0=train_cov, modulation="vector", return_std=sigma[macro_tickers], bekk_scale=100.0, sigma0_in_bekk_scale=False, gate_cov_layernorm=True, init_c_from_sigma0=True, jitter=bekk_jitter)
bekk_kernel_lstm_vec, hist_bekk_vec = train_covariance_model(
    bekk_kernel_lstm_vec,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss_bekk_vec,
    loss_kwargs=None,
    epochs=500,
    lr=bekk_lr,
    plateau_patience=20,
    device=bekk_device,
    scheduler_type="cosine",  # oder "cosine"
    grad_clip_max_norm=0.5,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
)

save_model_checkpoint(
    model_name="bekk_lstm_vector",
    models_dir=models_dir,
    model=bekk_kernel_lstm_vec,
    history=hist_bekk_vec,
    run_metadata=RUN_METADATA,
    loss_fn=student_loss_bekk_vec,
    config={
        "model_class": "BEKKLSTM",
        "distribution": "student_t",
        "modulation": "vector",
        "asym": True,
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": hidden_size,
        "lookback": lookback,
        "batchsize": batchsize,
        "epochs": 500,
        "lr": bekk_lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "device": bekk_device,
        "bekk_scale": 100.0,
        "jitter": bekk_jitter,
        "sigma0_in_bekk_scale": False,
        "gate_cov_layernorm": True,
        "init_c_from_sigma0": True,
        "nu": float(student_loss_bekk_vec.nu.detach().cpu()),
    },
)


# In[947]:


learned_nu_bekk_vec = float(student_loss_bekk_vec.nu.detach().cpu())
bekk_vec_student_kwargs = {"nu": learned_nu_bekk_vec}


# In[948]:


learned_nu_bekk_vec, learned_nu_bekk_vec


# In[949]:


plot_loss(hist_bekk_vec)


# In[950]:


sigma_val_scaled_bekk_vec, y_val_scaled_bekk_vec = predict_sigma(bekk_kernel_lstm_vec, vol_val_loader, device=bekk_device)
sigma_val_real_bekk_vec = scale_back(sigma_val_scaled_bekk_vec, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk_vec, y_test_scaled_bekk_vec = predict_sigma(bekk_kernel_lstm_vec, vol_test_loader, device=bekk_device)
sigma_test_real_bekk_vec = scale_back(sigma_test_scaled_bekk_vec, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk_vec, y_train_scaled_bekk_vec = predict_sigma(bekk_kernel_lstm_vec, vol_train_eval_loader, device=bekk_device)
sigma_train_real_bekk_vec = scale_back(sigma_train_scaled_bekk_vec, sigma=sigma, feature_cols=macro_tickers)


# In[951]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel)", df=train_df, pred_vol=sigma_train_real_bekk_vec, loss_fn=student_nll, loss_kwargs=bekk_vec_student_kwargs)


# In[952]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel)", df=val_df, pred_vol=sigma_val_real_bekk_vec, loss_fn=student_nll, loss_kwargs=bekk_vec_student_kwargs)


# In[953]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel)", df=test_df, pred_vol=sigma_test_real_bekk_vec, loss_fn=student_nll, loss_kwargs=bekk_vec_student_kwargs)


# ### Portfolio

# In[954]:


idx_test_bekk_vec, rp_test_bekk_vec, varp_test_bekk_vec, volp_test_bekk_vec = calc_portfolio_variance(
    test_df, sigma_test_real_bekk_vec, macro_tickers, lookback, w_np
)

idx_val_bekk_vec, rp_val_bekk_vec, varp_val_bekk_vec, volp_val_bekk_vec = calc_portfolio_variance(
    val_df, sigma_val_real_bekk_vec, macro_tickers, lookback, w_np
)

idx_train_bekk_vec, rp_train_bekk_vec, varp_train_bekk_vec, volp_train_bekk_vec = calc_portfolio_variance(
    train_df, sigma_train_real_bekk_vec, macro_tickers, lookback, w_np
)


# In[955]:


r = np.asarray(rp_test_bekk_vec, dtype=float)
v = np.asarray(volp_test_bekk_vec, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk_vec, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk_vec, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[956]:


portfolio_test_bekk_vec = make_portfolio_df(idx_test_bekk_vec, rp_test_bekk_vec, varp_test_bekk_vec, volp_test_bekk_vec)
portfolio_val_bekk_vec = make_portfolio_df(idx_val_bekk_vec, rp_val_bekk_vec, varp_val_bekk_vec, volp_val_bekk_vec)
portfolio_train_bekk_vec = make_portfolio_df(idx_train_bekk_vec, rp_train_bekk_vec, varp_train_bekk_vec, volp_train_bekk_vec)


# ### FHS

# In[957]:


z_train_bekk_vec = portfolio_train_bekk_vec["r_p"].values / np.clip(portfolio_train_bekk_vec["vol_p"].values, 1e-12, None) # clip sichert nur nach unten ab (0 division)
z_val_bekk_vec = portfolio_val_bekk_vec["r_p"].values / np.clip(portfolio_val_bekk_vec["vol_p"].values, 1e-12, None)
z_test_bekk_vec = portfolio_test_bekk_vec["r_p"].values / np.clip(portfolio_test_bekk_vec["vol_p"].values, 1e-12, None)
z_train_val_bekk_vec =np.concatenate([z_train_bekk_vec, z_val_bekk_vec])


var_fhs_test_bekk_vec, es_fhs_test_bekk_vec, hits_test_bekk_vec = fhs_var_es(
    z_hist_init=z_train_val_bekk_vec, # nimmt die letzen #window Werte als Historie
    r_oos=portfolio_test_bekk_vec["r_p"].values,
    vol_oos=portfolio_test_bekk_vec["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_test_bekk_vec = hits_test_bekk_vec.mean()
print(f"FHS Test hit rate = {hit_rate_test_bekk_vec:.3%} (target {alpha:.3%})")


# In[958]:


len(close)


# ### Backtesting

# !!! Backtestings müssen mit dem selben $\alpha$ vorgenommen werden wie mit FHS !!!

# Backtests sind für univariate Zeitreihen definiert, d.h. wir testen alle Modelle im Kontext eines vordefinierten Portfolios.

# In[959]:


results_bekk_vec = run_backtest_suite(
    returns=portfolio_test_bekk_vec["r_p"].values,
    var=var_fhs_test_bekk_vec,
    es=es_fhs_test_bekk_vec,
    alpha=alpha,
    volatility=portfolio_test_bekk_vec["vol_p"].values,
)

results_bekk_vec


# ## Convex Mixture Modulation

# In[960]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_bekk_mix = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[961]:


bekk_kernel_lstm_mix = BEKKLSTM(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=hidden_size, asym=True, Sigma0=train_cov, modulation="convex_mixture", return_std=sigma[macro_tickers], bekk_scale=100.0, sigma0_in_bekk_scale=False, gate_cov_layernorm=True, init_c_from_sigma0=True, jitter=bekk_jitter)
bekk_kernel_lstm_mix, hist_bekk_mix = train_covariance_model(
    bekk_kernel_lstm_mix,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss_bekk_mix,
    loss_kwargs=None,
    epochs=500,
    lr=bekk_mix_lr,
    plateau_patience=20,
    device=bekk_device,
    scheduler_type="cosine",  # oder "cosine"
    grad_clip_max_norm=0.5,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
)

save_model_checkpoint(
    model_name="bekk_lstm_mix",
    models_dir=models_dir,
    model=bekk_kernel_lstm_mix,
    history=hist_bekk_mix,
    run_metadata=RUN_METADATA,
    loss_fn=student_loss_bekk_mix,
    config={
        "model_class": "BEKKLSTM",
        "distribution": "student_t",
        "modulation": "convex_mixture",
        "asym": True,
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": hidden_size,
        "lookback": lookback,
        "batchsize": batchsize,
        "epochs": 500,
        "lr": bekk_mix_lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "device": bekk_device,
        "bekk_scale": 100.0,
        "jitter": bekk_jitter,
        "sigma0_in_bekk_scale": False,
        "gate_cov_layernorm": True,
        "convex_mixture_nn_on_bekk_scale": True,
        "init_c_from_sigma0": True,
        "nu": float(student_loss_bekk_mix.nu.detach().cpu()),
    },
)


# In[962]:


learned_nu_bekk_mix = float(student_loss_bekk_mix.nu.detach().cpu())
bekk_mix_student_kwargs = {"nu": learned_nu_bekk_mix}


# In[963]:


learned_nu_bekk_vec, learned_nu_bekk_mix


# In[964]:


plot_loss(hist_bekk_mix)


# In[965]:


sigma_val_scaled_bekk_mix, y_val_scaled_bekk_mix = predict_sigma(bekk_kernel_lstm_mix, vol_val_loader, device=bekk_device)
sigma_val_real_bekk_mix = scale_back(sigma_val_scaled_bekk_mix, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk_mix, y_test_scaled_bekk_mix = predict_sigma(bekk_kernel_lstm_mix, vol_test_loader, device=bekk_device)
sigma_test_real_bekk_mix = scale_back(sigma_test_scaled_bekk_mix, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk_mix, y_train_scaled_bekk_mix = predict_sigma(bekk_kernel_lstm_mix, vol_train_eval_loader, device=bekk_device)
sigma_train_real_bekk_mix = scale_back(sigma_train_scaled_bekk_mix, sigma=sigma, feature_cols=macro_tickers)


# In[966]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel Convex Mixture)", df=train_df, pred_vol=sigma_train_real_bekk_mix, loss_fn=student_nll, loss_kwargs=bekk_mix_student_kwargs)


# In[967]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel Convex Mixture)", df=val_df, pred_vol=sigma_val_real_bekk_mix, loss_fn=student_nll, loss_kwargs=bekk_mix_student_kwargs)


# In[968]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel Convex Mixture)", df=test_df, pred_vol=sigma_test_real_bekk_mix, loss_fn=student_nll, loss_kwargs=bekk_mix_student_kwargs)


# ### Portfolio

# In[969]:


idx_test_bekk_mix, rp_test_bekk_mix, varp_test_bekk_mix, volp_test_bekk_mix = calc_portfolio_variance(
    test_df, sigma_test_real_bekk_mix, macro_tickers, lookback, w_np
)

idx_val_bekk_mix, rp_val_bekk_mix, varp_val_bekk_mix, volp_val_bekk_mix = calc_portfolio_variance(
    val_df, sigma_val_real_bekk_mix, macro_tickers, lookback, w_np
)

idx_train_bekk_mix, rp_train_bekk_mix, varp_train_bekk_mix, volp_train_bekk_mix = calc_portfolio_variance(
    train_df, sigma_train_real_bekk_mix, macro_tickers, lookback, w_np
)


# In[970]:


r = np.asarray(rp_test_bekk_mix, dtype=float)
v = np.asarray(volp_test_bekk_mix, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk_mix, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk_mix, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[971]:


portfolio_test_bekk_mix = make_portfolio_df(idx_test_bekk_mix, rp_test_bekk_mix, varp_test_bekk_mix, volp_test_bekk_mix)
portfolio_val_bekk_mix = make_portfolio_df(idx_val_bekk_mix, rp_val_bekk_mix, varp_val_bekk_mix, volp_val_bekk_mix)
portfolio_train_bekk_mix = make_portfolio_df(idx_train_bekk_mix, rp_train_bekk_mix, varp_train_bekk_mix, volp_train_bekk_mix)


# ### FHS

# In[972]:


z_train_bekk_mix = portfolio_train_bekk_mix["r_p"].values / np.clip(portfolio_train_bekk_mix["vol_p"].values, 1e-12, None) # clip sichert nur nach unten ab (0 division)
z_val_bekk_mix = portfolio_val_bekk_mix["r_p"].values / np.clip(portfolio_val_bekk_mix["vol_p"].values, 1e-12, None)
z_test_bekk_mix = portfolio_test_bekk_mix["r_p"].values / np.clip(portfolio_test_bekk_mix["vol_p"].values, 1e-12, None)
z_train_val_bekk_mix =np.concatenate([z_train_bekk_mix, z_val_bekk_mix])


var_fhs_test_bekk_mix, es_fhs_test_bekk_mix, hits_test_bekk_mix = fhs_var_es(
    z_hist_init=z_train_val_bekk_mix, # nimmt die letzen #window Werte als Historie
    r_oos=portfolio_test_bekk_mix["r_p"].values,
    vol_oos=portfolio_test_bekk_mix["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_test_bekk_mix = hits_test_bekk_mix.mean()
print(f"FHS Test hit rate = {hit_rate_test_bekk_mix:.3%} (target {alpha:.3%})")


# ### Backtesting

# !!! Backtestings müssen mit dem selben $\alpha$ vorgenommen werden wie mit FHS !!!

# Backtests sind für univariate Zeitreihen definiert, d.h. wir testen alle Modelle im Kontext eines vordefinierten Portfolios.

# In[973]:


results_bekk_mix = run_backtest_suite(
    returns=portfolio_test_bekk_mix["r_p"].values,
    var=var_fhs_test_bekk_mix,
    es=es_fhs_test_bekk_mix,
    alpha=alpha,
    volatility=portfolio_test_bekk_mix["vol_p"].values,
)

results_bekk_mix


# # Neural-BEKK models

# **Deterministic Neural-BEKK**
# 
# $$\gamma_t = \left(\operatorname{vech}(C_t)^\top, \operatorname{vec}(A_t)^\top, \operatorname{vec}(G_t)^\top, \operatorname{vec}(B_t)^\top \right)^\top = f_\theta(\mathcal F_{t-1})$$
# 
# $$H_t =
# C_t C_t^\top
# +
# A_t^\top r_{t-1}r_{t-1}^\top A_t
# +
# G_t^\top \eta_{t-1}\eta_{t-1}^\top G_t
# +
# B_t^\top H_{t-1}B_t$$
# 
# $$r_t \mid \mathcal F_{t-1}
# \sim
# t_\nu(0,H_t)$$
# 
# **Probabilistic Neural-BEKK with stochastic \(C_t\)**
# 
# $$\mu_{C,t},\; s_{C,t},\; A_t,\; G_t,\; B_t =
# f_\theta(\mathcal F_{t-1})$$
# 
# $$c_t =
# \operatorname{vech}(C_t),
# \qquad
# c_t \mid \mathcal F_{t-1}
# \sim
# \mathcal N
# \left(
# \mu_{C,t},
# \operatorname{diag}(s_{C,t}^2)
# \right)$$
# 
# $$H_t^{(m)} = C_t^{(m)} C_t^{(m)\top} + A_t^\top r_{t-1}r_{t-1}^\top A_t + G_t^\top \eta_{t-1}\eta_{t-1}^\top G_t + B_t^\top H_{t-1}B_t$$
# 
# $$r_t \mid C_t^{(m)},\mathcal F_{t-1}
# \sim
# t_\nu(0,H_t^{(m)})$$
# 
# $$p(r_t \mid \mathcal F_{t-1})
# \approx
# \frac{1}{M}
# \sum_{m=1}^{M}
# t_\nu
# \left(
# r_t;0,H_t^{(m)}
# \right)$$
# 
# $$\mathcal L_t = -
# \log
# \left[
# \frac{1}{M}
# \sum_{m=1}^{M}
# t_\nu
# \left(
# r_t;0,H_t^{(m)}
# \right)
# \right]$$

# ## Deterministic

# In[99]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_neural_bekk = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[98]:


neural_bekk_asym_diag = NeuralBekk(
    n_assets=len(macro_tickers),
    input_size=(len(macro_tickers) + k),
    hidden_size=hidden_size,
    num_layers=1,
    dropout=dropout,
    bekk_type="diag",
    asym=True,
    Sigma0=train_cov,
    return_std=sigma[macro_tickers],
    bekk_scale=100.0,
    sigma0_in_bekk_scale=False,
    tau_max=0.995,
    c_delta_scale=0.01,
    init_C_from_sigma0=True,
    init_a=0.05,
    init_g=0.90,
    init_b=0.05,
    jitter=bekk_jitter,
)
neural_bekk_asym_diag, hist_neural_bekk = train_covariance_model(
    neural_bekk_asym_diag,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss_neural_bekk,
    loss_kwargs=None,
    epochs=500,
    lr=neural_bekk_lr,
    plateau_patience=20,
    device=bekk_device,
    scheduler_type="cosine",
    grad_clip_max_norm=0.5,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
)

save_model_checkpoint(
    model_name="neural_bekk_asym_diag",
    models_dir=models_dir,
    model=neural_bekk_asym_diag,
    history=hist_neural_bekk,
    run_metadata=RUN_METADATA,
    loss_fn=student_loss_neural_bekk,
    config={
        "model_class": "NeuralBekk",
        "distribution": "student_t",
        "bekk_type": "diag",
        "asym": True,
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": hidden_size,
        "num_layers": 1,
        "dropout": dropout,
        "lookback": lookback,
        "batchsize": batchsize,
        "epochs": 500,
        "lr": neural_bekk_lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "device": bekk_device,
        "bekk_scale": 100.0,
        "jitter": bekk_jitter,
        "sigma0_in_bekk_scale": False,
        "tau_max": 0.995,
        "c_delta_scale": 0.01,
        "init_C_from_sigma0": True,
        "init_a": 0.05,
        "recursion_jitter": 0.0,
        "cholesky_jitter": bekk_jitter,
        "init_g": 0.90,
        "init_b": 0.05,
        "nu": float(student_loss_neural_bekk.nu.detach().cpu()),
    },
)


# In[ ]:


learned_nu_neural_bekk = float(student_loss_neural_bekk.nu.detach().cpu())
neural_bekk_student_kwargs = {"nu": learned_nu_neural_bekk}
learned_nu_neural_bekk


# In[ ]:


plot_loss(hist_neural_bekk)


# In[ ]:


sigma_val_scaled_neural_bekk, y_val_scaled_neural_bekk = predict_sigma(neural_bekk_asym_diag, vol_val_loader, device=bekk_device)
sigma_val_real_neural_bekk = scale_back(sigma_val_scaled_neural_bekk, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_neural_bekk, y_test_scaled_neural_bekk = predict_sigma(neural_bekk_asym_diag, vol_test_loader, device=bekk_device)
sigma_test_real_neural_bekk = scale_back(sigma_test_scaled_neural_bekk, sigma=sigma, feature_cols=macro_tickers)

sigma_train_scaled_neural_bekk, y_train_scaled_neural_bekk = predict_sigma(neural_bekk_asym_diag, vol_train_eval_loader, device=bekk_device)
sigma_train_real_neural_bekk = scale_back(sigma_train_scaled_neural_bekk, sigma=sigma, feature_cols=macro_tickers)


# In[ ]:


def predict_neural_bekk_params(model, loader, device="cpu"):
    model.eval()
    collected = {
        "C": [],
        "A": [],
        "G": [],
        "B": [],
        "a": [],
        "g": [],
        "b": [],
        "tau": [],
        "weights": [],
    }

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, _, params = model(xb, return_params=True)
            for key in collected:
                if params[key] is not None:
                    collected[key].append(params[key].cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in collected.items() if value}


neural_bekk_test_params = predict_neural_bekk_params(
    neural_bekk_asym_diag,
    vol_test_loader,
    device=bekk_device,
)

neural_bekk_param_npz_path = run_dir / "neural_bekk_asym_diag_test_param_sequences.npz"
np.savez_compressed(neural_bekk_param_npz_path, **neural_bekk_test_params)

neural_bekk_param_columns = {}
for param_name in ("a", "g", "b", "tau"):
    last_step_values = neural_bekk_test_params[param_name][:, -1, :]
    for asset_idx, ticker in enumerate(macro_tickers):
        neural_bekk_param_columns[f"{param_name}_{ticker}"] = last_step_values[:, asset_idx]

last_weights = neural_bekk_test_params["weights"][:, -1, :, :]
for weight_idx, weight_name in enumerate(("a", "g", "b")):
    for asset_idx, ticker in enumerate(macro_tickers):
        neural_bekk_param_columns[f"weight_{weight_name}_{ticker}"] = last_weights[:, asset_idx, weight_idx]

neural_bekk_param_df = pd.DataFrame(
    neural_bekk_param_columns,
    index=test_df.index[lookback:],
)
neural_bekk_param_df.index.name = "Date"
neural_bekk_param_path = run_dir / "neural_bekk_asym_diag_test_params.csv"
neural_bekk_param_df.to_csv(neural_bekk_param_path)
neural_bekk_param_df.head()


# In[ ]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + Neural Asymmetric Diagonal BEKK)", df=train_df, pred_vol=sigma_train_real_neural_bekk, loss_fn=student_nll, loss_kwargs=neural_bekk_student_kwargs)


# In[ ]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + Neural Asymmetric Diagonal BEKK)", df=val_df, pred_vol=sigma_val_real_neural_bekk, loss_fn=student_nll, loss_kwargs=neural_bekk_student_kwargs)


# In[ ]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + Neural Asymmetric Diagonal BEKK)", df=test_df, pred_vol=sigma_test_real_neural_bekk, loss_fn=student_nll, loss_kwargs=neural_bekk_student_kwargs)


# ### Portfolio

# In[ ]:


idx_test_neural_bekk, rp_test_neural_bekk, varp_test_neural_bekk, volp_test_neural_bekk = calc_portfolio_variance(
    test_df, sigma_test_real_neural_bekk, macro_tickers, lookback, w_np
)

idx_val_neural_bekk, rp_val_neural_bekk, varp_val_neural_bekk, volp_val_neural_bekk = calc_portfolio_variance(
    val_df, sigma_val_real_neural_bekk, macro_tickers, lookback, w_np
)

idx_train_neural_bekk, rp_train_neural_bekk, varp_train_neural_bekk, volp_train_neural_bekk = calc_portfolio_variance(
    train_df, sigma_train_real_neural_bekk, macro_tickers, lookback, w_np
)


# In[ ]:


r = np.asarray(rp_test_neural_bekk, dtype=float)
v = np.asarray(volp_test_neural_bekk, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_neural_bekk, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_neural_bekk, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


portfolio_test_neural_bekk = make_portfolio_df(idx_test_neural_bekk, rp_test_neural_bekk, varp_test_neural_bekk, volp_test_neural_bekk)
portfolio_val_neural_bekk = make_portfolio_df(idx_val_neural_bekk, rp_val_neural_bekk, varp_val_neural_bekk, volp_val_neural_bekk)
portfolio_train_neural_bekk = make_portfolio_df(idx_train_neural_bekk, rp_train_neural_bekk, varp_train_neural_bekk, volp_train_neural_bekk)


# ### FHS

# In[ ]:


z_train_neural_bekk = portfolio_train_neural_bekk["r_p"].values / np.clip(portfolio_train_neural_bekk["vol_p"].values, 1e-12, None)
z_val_neural_bekk = portfolio_val_neural_bekk["r_p"].values / np.clip(portfolio_val_neural_bekk["vol_p"].values, 1e-12, None)
z_test_neural_bekk = portfolio_test_neural_bekk["r_p"].values / np.clip(portfolio_test_neural_bekk["vol_p"].values, 1e-12, None)
z_train_val_neural_bekk = np.concatenate([z_train_neural_bekk, z_val_neural_bekk])

var_fhs_test_neural_bekk, es_fhs_test_neural_bekk, hits_test_neural_bekk = fhs_var_es(
    z_hist_init=z_train_val_neural_bekk,
    r_oos=portfolio_test_neural_bekk["r_p"].values,
    vol_oos=portfolio_test_neural_bekk["vol_p"].values,
    alpha=alpha,
    window=window,
)

hit_rate_test_neural_bekk = hits_test_neural_bekk.mean()
print(f"FHS Test hit rate = {hit_rate_test_neural_bekk:.3%} (target {alpha:.3%})")


# ### Backtesting

# In[ ]:


results_neural_bekk_asym_diag = run_backtest_suite(
    returns=portfolio_test_neural_bekk["r_p"].values,
    var=var_fhs_test_neural_bekk,
    es=es_fhs_test_neural_bekk,
    alpha=alpha,
    volatility=portfolio_test_neural_bekk["vol_p"].values,
)

results_neural_bekk_asym_diag


# ## Probablistic

# In[ ]:





# # GRU for Covariance Matrix

# In[974]:


from GRUCovariance import GRUCovariance


# In[975]:


# GRU Hyperparameters
gru_lookback = lookback
gru_batchsize = 128
gru_hidden_size = 64
gru_num_layers = 2
gru_dropout = 0.2
gru_epochs = 500
gru_lr = 1e-4

if gru_lookback != lookback:
    raise ValueError(
        "GRU uses the existing SequenceDataset objects. "
        "Keep gru_lookback == lookback or rebuild the datasets/loaders first."
    )


# ## Helper Functions
# 

# In[976]:


gru_split_frames = {
    "train": train_df,
    "val": val_df,
    "test": test_df,
}

gru_eval_loaders = {
    "train": DataLoader(vol_train_ds, batch_size=gru_batchsize, shuffle=False),
    "val": DataLoader(vol_val_ds, batch_size=gru_batchsize, shuffle=False),
    "test": DataLoader(vol_test_ds, batch_size=gru_batchsize, shuffle=False),
}

def fit_gru_model(loss_fn=gaussian_nll, loss_kwargs=None, epochs=gru_epochs, lr=gru_lr):
    k = len(macro_tickers) * (len(macro_tickers) + 1) // 2
    seed_everything(SEED)
    gru_train_loader = DataLoader(
        vol_train_ds,
        batch_size=gru_batchsize,
        shuffle=True,
        generator=make_generator(SEED),
    )
    model = GRUCovariance(
        input_size=len(macro_tickers) + k,
        n_assets=len(macro_tickers),
        hidden_size=gru_hidden_size,
        num_layers=gru_num_layers,
        dropout=gru_dropout,
    )

    model, history = train_covariance_model(
        model,
        gru_train_loader,
        gru_eval_loaders["val"],
        loss_fn=loss_fn,
        loss_kwargs=loss_kwargs,
        epochs=epochs,
        lr=lr,
        plateau_patience=20,
        device=device,
        scheduler_type="cosine",
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    sigma_train_scaled, y_train_scaled = predict_sigma(model, gru_eval_loaders["train"], device=device)
    sigma_val_scaled, y_val_scaled = predict_sigma(model, gru_eval_loaders["val"], device=device)
    sigma_test_scaled, y_test_scaled = predict_sigma(model, gru_eval_loaders["test"], device=device)

    sigma_splits = {
        "train": scale_back(sigma_train_scaled, sigma=sigma, feature_cols=macro_tickers),
        "val": scale_back(sigma_val_scaled, sigma=sigma, feature_cols=macro_tickers),
        "test": scale_back(sigma_test_scaled, sigma=sigma, feature_cols=macro_tickers),
    }
    y_splits = {
        "train": y_train_scaled,
        "val": y_val_scaled,
        "test": y_test_scaled,
    }
    return model, history, sigma_splits, y_splits

def build_gru_portfolio_splits(sigma_splits, w):
    portfolio_splits = {}
    for split_name, df_split in gru_split_frames.items():
        idx, rp, varp, volp = calc_portfolio_variance(
            df_split,
            sigma_splits[split_name],
            macro_tickers,
            gru_lookback,
            w,
        )
        portfolio_splits[split_name] = make_portfolio_df(idx, rp, varp, volp)
    return portfolio_splits

def standardized_portfolio_returns(portfolio_df):
    return portfolio_df["r_p"].values / np.clip(portfolio_df["vol_p"].values, 1e-12, None)

def run_fhs_pipeline(portfolio_train, portfolio_val, portfolio_test, alpha=0.01, window=1000):
    z_train = standardized_portfolio_returns(portfolio_train)
    z_val = standardized_portfolio_returns(portfolio_val)
    z_test = standardized_portfolio_returns(portfolio_test)
    z_train_val = np.concatenate([z_train, z_val])

    train_burn_in = min(window, len(z_train) - 1)
    if train_burn_in < 1:
        raise ValueError("Training split is too short for FHS burn-in.")

    var_fhs_test, es_fhs_test, hits_test = fhs_var_es(
        z_hist_init=z_train_val,
        r_oos=portfolio_test["r_p"].values,
        vol_oos=portfolio_test["vol_p"].values,
        alpha=alpha,
        window=window,
    )
    var_fhs_val, es_fhs_val, hit_val = fhs_var_es(
        z_hist_init=z_train,
        r_oos=portfolio_val["r_p"].values,
        vol_oos=portfolio_val["vol_p"].values,
        alpha=alpha,
        window=window,
    )
    var_fhs_train, es_fhs_train, hit_train = fhs_var_es(
        z_hist_init=z_train[:train_burn_in],
        r_oos=portfolio_train["r_p"].iloc[train_burn_in:].values,
        vol_oos=portfolio_train["vol_p"].iloc[train_burn_in:].values,
        alpha=alpha,
        window=window,
    )

    return {
        "z": {
            "train": z_train,
            "val": z_val,
            "test": z_test,
            "train_val": z_train_val,
        },
        "burn_in": train_burn_in,
        "train": {"var": var_fhs_train, "es": es_fhs_train, "hits": hit_train},
        "val": {"var": var_fhs_val, "es": es_fhs_val, "hits": hit_val},
        "test": {"var": var_fhs_test, "es": es_fhs_test, "hits": hits_test},
    }

def plot_fhs_var(portfolio_df, var_fhs, alpha, title, start=0):
    idx = portfolio_df.index[start:]
    returns = portfolio_df["r_p"].iloc[start:].values
    plt.figure(figsize=(12, 4))
    plt.plot(idx, returns, color="black", lw=0.8, label="Portfolio Return")
    plt.plot(idx, var_fhs, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
    plt.fill_between(idx, var_fhs, np.zeros_like(var_fhs), color="tab:red", alpha=0.08)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ## GRU with Gaussian NLL
# 

# In[977]:


gru_model, gru_hist, gru_sigma_splits, gru_y_splits = fit_gru_model(loss_fn=gaussian_nll)

gru_sigma_train_real = gru_sigma_splits["train"]
gru_sigma_val_real = gru_sigma_splits["val"]
gru_sigma_test_real = gru_sigma_splits["test"]

save_model_checkpoint(
    model_name="gru_normal",
    models_dir=models_dir,
    model=gru_model,
    history=gru_hist,
    run_metadata=RUN_METADATA,
    loss_fn=None,
    config={
        "model_class": "GRUCovariance",
        "distribution": "gaussian",
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": gru_hidden_size,
        "num_layers": gru_num_layers,
        "dropout": gru_dropout,
        "lookback": gru_lookback,
        "batchsize": gru_batchsize,
        "epochs": gru_epochs,
        "lr": gru_lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    },
)

plot_loss(gru_hist)


# In[978]:


plot_var(alpha=0.01, lookback=gru_lookback, cols=macro_tickers, view="Train-Split (GRU Gaussian)", df=train_df, pred_vol=gru_sigma_train_real)
plot_var(alpha=0.01, lookback=gru_lookback, cols=macro_tickers, view="Validation-Split (GRU Gaussian)", df=val_df, pred_vol=gru_sigma_val_real)
plot_var(alpha=0.01, lookback=gru_lookback, cols=macro_tickers, view="Test-Split (GRU Gaussian)", df=test_df, pred_vol=gru_sigma_test_real)


# ## Portfolio (Gaussian NLL)
# 

# In[979]:


portfolio_logrets = torch.tensor(logret[macro_tickers].values, dtype=torch.float32, device=device) @ w
portfolio_logrets_np = portfolio_logrets.detach().cpu().numpy()

plt.figure(figsize=(12, 4))
plt.plot(logret.index, portfolio_logrets_np, label="Portfolio Log-Returns")
plt.title("Equal-Weight Portfolio Log-Returns")
plt.legend()
plt.tight_layout()
plt.show()

portfolio_splits_gru = build_gru_portfolio_splits(gru_sigma_splits, w_np)
portfolio_train_gru = portfolio_splits_gru["train"]
portfolio_val_gru = portfolio_splits_gru["val"]
portfolio_test_gru = portfolio_splits_gru["test"]

portfolio_test_gru.head()


# In[980]:


plot_var(alpha=0.01, lookback=gru_lookback, view="Train-Split (GRU Gaussian)", portfolio=True, portfolio_df=portfolio_train_gru)
plot_var(alpha=0.01, lookback=gru_lookback, view="Validation-Split (GRU Gaussian)", portfolio=True, portfolio_df=portfolio_val_gru)
plot_var(alpha=0.01, lookback=gru_lookback, view="Test-Split (GRU Gaussian)", portfolio=True, portfolio_df=portfolio_test_gru)


# ## GRU with Student-t NLL
# 

# In[981]:


gru_student_loss = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)
gru_model_t, gru_hist_t, gru_sigma_splits_t, gru_y_splits_t = fit_gru_model(loss_fn=gru_student_loss)

gru_learned_nu = float(gru_student_loss.nu.detach().cpu())
gru_student_kwargs = {"nu": gru_learned_nu}

gru_sigma_train_real_t = gru_sigma_splits_t["train"]
gru_sigma_val_real_t = gru_sigma_splits_t["val"]
gru_sigma_test_real_t = gru_sigma_splits_t["test"]

save_model_checkpoint(
    model_name="gru_student",
    models_dir=models_dir,
    model=gru_model_t,
    history=gru_hist_t,
    run_metadata=RUN_METADATA,
    loss_fn=gru_student_loss,
    config={
        "model_class": "GRUCovariance",
        "distribution": "student_t",
        "input_size": len(macro_tickers) + k,
        "n_assets": len(macro_tickers),
        "hidden_size": gru_hidden_size,
        "num_layers": gru_num_layers,
        "dropout": gru_dropout,
        "lookback": gru_lookback,
        "batchsize": gru_batchsize,
        "epochs": gru_epochs,
        "lr": gru_lr,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "nu": gru_learned_nu,
    },
)

gru_learned_nu


# In[982]:


plot_loss(gru_hist_t)


# In[983]:


plot_var(alpha=0.01, lookback=gru_lookback, cols=macro_tickers, view="Train-Split (GRU Student-t)", df=train_df, pred_vol=gru_sigma_train_real_t, loss_fn=student_nll, loss_kwargs=gru_student_kwargs)
plot_var(alpha=0.01, lookback=gru_lookback, cols=macro_tickers, view="Validation-Split (GRU Student-t)", df=val_df, pred_vol=gru_sigma_val_real_t, loss_fn=student_nll, loss_kwargs=gru_student_kwargs)
plot_var(alpha=0.01, lookback=gru_lookback, cols=macro_tickers, view="Test-Split (GRU Student-t)", df=test_df, pred_vol=gru_sigma_test_real_t, loss_fn=student_nll, loss_kwargs=gru_student_kwargs)


# ## Portfolio (Student-t NLL)
# 

# In[984]:


portfolio_splits_gru_t = build_gru_portfolio_splits(gru_sigma_splits_t, w_np)
portfolio_train_gru_t = portfolio_splits_gru_t["train"]
portfolio_val_gru_t = portfolio_splits_gru_t["val"]
portfolio_test_gru_t = portfolio_splits_gru_t["test"]

portfolio_test_gru_t.head()


# In[985]:


plot_var(alpha=0.01, lookback=gru_lookback, view="Train-Split (GRU Student-t)", portfolio=True, portfolio_df=portfolio_train_gru_t, loss_fn=student_nll, loss_kwargs=gru_student_kwargs)
plot_var(alpha=0.01, lookback=gru_lookback, view="Validation-Split (GRU Student-t)", portfolio=True, portfolio_df=portfolio_val_gru_t, loss_fn=student_nll, loss_kwargs=gru_student_kwargs)
plot_var(alpha=0.01, lookback=gru_lookback, view="Test-Split (GRU Student-t)", portfolio=True, portfolio_df=portfolio_test_gru_t, loss_fn=student_nll, loss_kwargs=gru_student_kwargs)


# ## Filtered Historical Simulation
# 

# In[986]:


alpha = 0.01
window = 1000

fhs_gru_normal = run_fhs_pipeline(portfolio_train_gru, portfolio_val_gru, portfolio_test_gru, alpha=alpha, window=window)
var_fhs_train_gru = fhs_gru_normal["train"]["var"]
es_fhs_train_gru = fhs_gru_normal["train"]["es"]
hit_train_gru = fhs_gru_normal["train"]["hits"]

var_fhs_val_gru = fhs_gru_normal["val"]["var"]
es_fhs_val_gru = fhs_gru_normal["val"]["es"]
hit_val_gru = fhs_gru_normal["val"]["hits"]

var_fhs_test_gru = fhs_gru_normal["test"]["var"]
es_fhs_test_gru = fhs_gru_normal["test"]["es"]
hits_test_gru = fhs_gru_normal["test"]["hits"]

pd.DataFrame(
    {
        "hit_rate": [hit_train_gru.mean(), hit_val_gru.mean(), hits_test_gru.mean()],
    },
    index=["train", "val", "test"],
)


# In[987]:


plot_fhs_var(
    portfolio_train_gru,
    var_fhs_train_gru,
    alpha=alpha,
    title="Filtered Historical Simulation VaR (GRU Gaussian, Train)",
    start=fhs_gru_normal["burn_in"],
)
plot_fhs_var(
    portfolio_val_gru,
    var_fhs_val_gru,
    alpha=alpha,
    title="Filtered Historical Simulation VaR (GRU Gaussian, Validation)",
)
plot_fhs_var(
    portfolio_test_gru,
    var_fhs_test_gru,
    alpha=alpha,
    title="Filtered Historical Simulation VaR (GRU Gaussian, Test)",
)


# In[988]:


fhs_gru_student = run_fhs_pipeline(portfolio_train_gru_t, portfolio_val_gru_t, portfolio_test_gru_t, alpha=alpha, window=window)
var_fhs_train_gru_t = fhs_gru_student["train"]["var"]
es_fhs_train_gru_t = fhs_gru_student["train"]["es"]
hit_train_gru_t = fhs_gru_student["train"]["hits"]

var_fhs_val_gru_t = fhs_gru_student["val"]["var"]
es_fhs_val_gru_t = fhs_gru_student["val"]["es"]
hit_val_gru_t = fhs_gru_student["val"]["hits"]

var_fhs_test_gru_t = fhs_gru_student["test"]["var"]
es_fhs_test_gru_t = fhs_gru_student["test"]["es"]
hits_test_gru_t = fhs_gru_student["test"]["hits"]

pd.DataFrame(
    {
        "hit_rate": [hit_train_gru_t.mean(), hit_val_gru_t.mean(), hits_test_gru_t.mean()],
    },
    index=["train", "val", "test"],
)


# In[989]:


plot_fhs_var(
    portfolio_train_gru_t,
    var_fhs_train_gru_t,
    alpha=alpha,
    title="Filtered Historical Simulation VaR (GRU Student-t, Train)",
    start=fhs_gru_student["burn_in"],
)
plot_fhs_var(
    portfolio_val_gru_t,
    var_fhs_val_gru_t,
    alpha=alpha,
    title="Filtered Historical Simulation VaR (GRU Student-t, Validation)",
)
plot_fhs_var(
    portfolio_test_gru_t,
    var_fhs_test_gru_t,
    alpha=alpha,
    title="Filtered Historical Simulation VaR (GRU Student-t, Test)",
)


# ## GRU Backtesting
# 

# In[990]:


from backtesting import build_fz_loss_matrix, run_backtest_suite, run_fz_dm_comparison_plan


# In[991]:


results_gru_normal = run_backtest_suite(
    returns=portfolio_test_gru["r_p"].values,
    var=var_fhs_test_gru,
    es=es_fhs_test_gru,
    alpha=alpha,
    volatility=portfolio_test_gru["vol_p"].values,
)

results_gru_student = run_backtest_suite(
    returns=portfolio_test_gru_t["r_p"].values,
    var=var_fhs_test_gru_t,
    es=es_fhs_test_gru_t,
    alpha=alpha,
    volatility=portfolio_test_gru_t["vol_p"].values,
)

{
    "gru_normal": results_gru_normal,
    "gru_student": results_gru_student,
}


# # Backtesting

# !!! Backtestings müssen mit dem selben $\alpha$ vorgenommen werden wie mit FHS !!!

# Backtests sind für univariate Zeitreihen definiert, d.h. wir testen alle Modelle im Kontext eines vordefinierten Portfolios.

# In[992]:


#importlib.reload(backtesting)


# In[993]:


from backtesting import build_fz_loss_matrix, run_backtest_suite, run_fz_dm_comparison_plan


# In[994]:


results_lstm_normal = run_backtest_suite(
    returns=portfolio_test["r_p"].values,
    var=var_fhs_test,
    es=es_fhs_test,
    alpha=alpha,
    volatility=portfolio_test["vol_p"].values,  # optional, aber für CC/ER sinnvoll
)

results_lstm_normal


# In[995]:


results_lstm_student = run_backtest_suite(
    returns=portfolio_test_t["r_p"].values,
    var=var_fhs_test_t,
    es=es_fhs_test_t,
    alpha=alpha,
    volatility=portfolio_test_t["vol_p"].values,
)

results_lstm_student


# In[996]:


results_bekk_scalar = run_backtest_suite(
    returns=portfolio_test_bekk["r_p"].values,
    var=var_fhs_test_bekk,
    es=es_fhs_test_bekk,
    alpha=alpha,
    volatility=portfolio_test_bekk["vol_p"].values,
)

results_bekk_scalar # BEKK-LSTM scalar modulation


# # BEKK(1, 1) symmetrisch & asymmetrisch
# 
# importiert aus R bzw BEKKs Library.

# ## Daten import

# In[997]:


import shutil
import subprocess


# In[998]:


rscript = shutil.which("Rscript")
if rscript is None:
    raise RuntimeError("Rscript wurde nicht gefunden. Prüfe, ob R im PATH liegt.")


# In[999]:


cmd = [
    rscript,
    str(project_dir / "r" / "bekks.R"),
    "--data-id", DATA_ID,
    "--run-id", RUN_ID,
    "--project-dir", str(project_dir),
    "--seed", str(SEED),
    "--max-iter", "200",
]

result = subprocess.run(
    cmd,
    cwd=project_dir,
    capture_output=True,
    text=True,
    check=False,
)

print("R stdout:")
print(result.stdout)

if result.stderr:
    print("R stderr:")
    print(result.stderr)

if result.returncode != 0:
    raise RuntimeError(f"bekks.R failed with return code {result.returncode}")

print(f"BEKK run finished: {RUN_ID}")


# In[1000]:


def load_covariances(path, d, model_label="covariance"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{model_label} forecast file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    expected_cols = [
        f"h_{i}{j}"
        for i in range(1, d + 1)
        for j in range(1, d + 1)
    ]

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing covariance columns in {path}: {missing_cols}")

    H = df[expected_cols].to_numpy(dtype=float).reshape(len(df), d, d)

    return df, H

load_bekk_covariances = load_covariances

sym_path = output_dir / f"bekk_forecasts_symmetric_{RUN_ID}.csv"
asym_path = output_dir / f"bekk_forecasts_asymmetric_{RUN_ID}.csv"

bekk_sym_df, H_bekk_sym = load_covariances(sym_path, d=len(macro_tickers), model_label="Symmetric BEKK")
bekk_asym_df, H_bekk_asym = load_covariances(asym_path, d=len(macro_tickers), model_label="Asymmetric BEKK")

if len(H_bekk_sym) != len(test_df):
    raise ValueError(f"Symmetric BEKK length mismatch: {len(H_bekk_sym)} vs test_df {len(test_df)}")

if len(H_bekk_asym) != len(test_df):
    raise ValueError(f"Asymmetric BEKK length mismatch: {len(H_bekk_asym)} vs test_df {len(test_df)}")

print("Symmetric BEKK:", H_bekk_sym.shape)
print("Asymmetric BEKK:", H_bekk_asym.shape)


# In[1001]:


sigma_test_real.shape


# BEKK(1, 1) hat ```lookback``` mehr OOS predictions als die LSTM Modelle

# In[1002]:


H_bekk_sym_aligned = H_bekk_sym[lookback:]
H_bekk_asym_aligned = H_bekk_asym[lookback:]

idx_bekk_sym, rp_bekk_sym, varp_bekk_sym, volp_bekk_sym = calc_portfolio_variance(
    test_df,
    H_bekk_sym_aligned,
    macro_tickers,
    lookback,
    w_np,
)

idx_bekk_asym, rp_bekk_asym, varp_bekk_asym, volp_bekk_asym = calc_portfolio_variance(
    test_df,
    H_bekk_asym_aligned,
    macro_tickers,
    lookback,
    w_np,
)


# In[1003]:


portfolio_test_bekk_sym = make_portfolio_df(idx_bekk_sym, rp_bekk_sym, varp_bekk_sym, volp_bekk_sym)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (BEKK Symmetric)", portfolio=True, portfolio_df=portfolio_test_bekk_sym)


# In[1004]:


portfolio_test_bekk_asym = make_portfolio_df(idx_bekk_asym, rp_bekk_asym, varp_bekk_asym, volp_bekk_asym)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (BEKK Asymmetric)", portfolio=True, portfolio_df=portfolio_test_bekk_asym)


# ## FHS

# In[1005]:


trainval_sym_path = output_dir / f"bekk_forecasts_fitted_train_val_symmetric_{RUN_ID}.csv"
trainval_asym_path = output_dir / f"bekk_forecasts_fitted_train_val_asymmetric_{RUN_ID}.csv"

_, H_bekk_sym_trainval = load_covariances(trainval_sym_path, d=len(macro_tickers), model_label="Symmetric BEKK train+val")
_, H_bekk_asym_trainval = load_covariances(trainval_asym_path, d=len(macro_tickers), model_label="Asymmetric BEKK train+val")


# In[1006]:


train_val_returns_df = pd.concat(
    [train_df[macro_tickers], val_df[macro_tickers]],
    axis=0,
)

idx_sym_hist, rp_sym_hist, varp_sym_hist, volp_sym_hist = calc_portfolio_variance(
    train_val_returns_df,
    H_bekk_sym_trainval,
    macro_tickers,
    0,
    w_np,
)

idx_asym_hist, rp_asym_hist, varp_asym_hist, volp_asym_hist = calc_portfolio_variance(
    train_val_returns_df,
    H_bekk_asym_trainval,
    macro_tickers,
    0,
    w_np,
)

portfolio_hist_bekk_sym = make_portfolio_df(idx_sym_hist, rp_sym_hist, varp_sym_hist, volp_sym_hist)
portfolio_hist_bekk_asym = make_portfolio_df(idx_asym_hist, rp_asym_hist, varp_asym_hist, volp_asym_hist)


# In[1007]:


z_hist_sym = portfolio_hist_bekk_sym["r_p"].values / np.clip(
    portfolio_hist_bekk_sym["vol_p"].values,
    1e-12,
    None,
)

z_hist_asym = portfolio_hist_bekk_asym["r_p"].values / np.clip(
    portfolio_hist_bekk_asym["vol_p"].values,
    1e-12,
    None,
)

var_fhs_test_bekk_sym, es_fhs_test_bekk_sym, hits_test_bekk_sym = fhs_var_es(
    z_hist_init=z_hist_sym,
    r_oos=portfolio_test_bekk_sym["r_p"].values,
    vol_oos=portfolio_test_bekk_sym["vol_p"].values,
    alpha=alpha,
    window=window,
)

var_fhs_test_bekk_asym, es_fhs_test_bekk_asym, hits_test_bekk_asym = fhs_var_es(
    z_hist_init=z_hist_asym,
    r_oos=portfolio_test_bekk_asym["r_p"].values,
    vol_oos=portfolio_test_bekk_asym["vol_p"].values,
    alpha=alpha,
    window=window,
)

print("Sym BEKK hit rate:", hits_test_bekk_sym.mean())
print("Asym BEKK hit rate:", hits_test_bekk_asym.mean())


# ## Backtest

# In[1008]:


results_bekk_sym = run_backtest_suite(
    returns=portfolio_test_bekk_sym["r_p"].values,
    var=var_fhs_test_bekk_sym,
    es=es_fhs_test_bekk_sym,
    alpha=alpha,
    volatility=portfolio_test_bekk_sym["vol_p"].values,
)

results_bekk_sym


# In[1009]:


results_bekk_asym = run_backtest_suite(
    returns=portfolio_test_bekk_asym["r_p"].values,
    var=var_fhs_test_bekk_asym,
    es=es_fhs_test_bekk_asym,
    alpha=alpha,
    volatility=portfolio_test_bekk_asym["vol_p"].values,
)

results_bekk_asym


# # DCC(1,1) & aDCC(1,1)
# 
# Importiert aus R via `rmgarch`.
# 

# ## R-Schätzung
# 

# In[1010]:


dcc_cmd = [
    rscript,
    str(project_dir / "r" / "dcc.R"),
    "--data-id", DATA_ID,
    "--run-id", RUN_ID,
    "--project-dir", str(project_dir),
    "--seed", str(SEED),
]

dcc_result = subprocess.run(
    dcc_cmd,
    cwd=project_dir,
    capture_output=True,
    text=True,
    check=False,
)

print("R stdout:")
print(dcc_result.stdout)

if dcc_result.stderr:
    print("R stderr:")
    print(dcc_result.stderr)

if dcc_result.returncode != 0:
    raise RuntimeError(f"dcc.R failed with return code {dcc_result.returncode}")

print(f"DCC run finished: {RUN_ID}")


# ## Forecasts
# 

# In[1011]:


dcc_path = output_dir / f"dcc_forecasts_dcc_{RUN_ID}.csv"
adcc_path = output_dir / f"dcc_forecasts_adcc_{RUN_ID}.csv"

dcc_df, H_dcc = load_covariances(dcc_path, d=len(macro_tickers), model_label="DCC")
adcc_df, H_adcc = load_covariances(adcc_path, d=len(macro_tickers), model_label="aDCC")

if len(H_dcc) != len(test_df):
    raise ValueError(f"DCC length mismatch: {len(H_dcc)} vs test_df {len(test_df)}")

if len(H_adcc) != len(test_df):
    raise ValueError(f"aDCC length mismatch: {len(H_adcc)} vs test_df {len(test_df)}")

print("DCC:", H_dcc.shape)
print("aDCC:", H_adcc.shape)


# In[1012]:


H_dcc_aligned = H_dcc[lookback:]
H_adcc_aligned = H_adcc[lookback:]

idx_dcc, rp_dcc, varp_dcc, volp_dcc = calc_portfolio_variance(
    test_df,
    H_dcc_aligned,
    macro_tickers,
    lookback,
    w_np,
)

idx_adcc, rp_adcc, varp_adcc, volp_adcc = calc_portfolio_variance(
    test_df,
    H_adcc_aligned,
    macro_tickers,
    lookback,
    w_np,
)


# In[1013]:


portfolio_test_dcc = make_portfolio_df(idx_dcc, rp_dcc, varp_dcc, volp_dcc)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (DCC)", portfolio=True, portfolio_df=portfolio_test_dcc)


# In[1014]:


portfolio_test_adcc = make_portfolio_df(idx_adcc, rp_adcc, varp_adcc, volp_adcc)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (aDCC)", portfolio=True, portfolio_df=portfolio_test_adcc)


# ## FHS
# 

# In[1015]:


trainval_dcc_path = output_dir / f"dcc_forecasts_fitted_train_val_dcc_{RUN_ID}.csv"
trainval_adcc_path = output_dir / f"dcc_forecasts_fitted_train_val_adcc_{RUN_ID}.csv"

_, H_dcc_trainval = load_covariances(trainval_dcc_path, d=len(macro_tickers), model_label="DCC train+val")
_, H_adcc_trainval = load_covariances(trainval_adcc_path, d=len(macro_tickers), model_label="aDCC train+val")


# In[1016]:


train_val_returns_df = pd.concat(
    [train_df[macro_tickers], val_df[macro_tickers]],
    axis=0,
)

idx_dcc_hist, rp_dcc_hist, varp_dcc_hist, volp_dcc_hist = calc_portfolio_variance(
    train_val_returns_df,
    H_dcc_trainval,
    macro_tickers,
    0,
    w_np,
)

idx_adcc_hist, rp_adcc_hist, varp_adcc_hist, volp_adcc_hist = calc_portfolio_variance(
    train_val_returns_df,
    H_adcc_trainval,
    macro_tickers,
    0,
    w_np,
)

portfolio_hist_dcc = make_portfolio_df(idx_dcc_hist, rp_dcc_hist, varp_dcc_hist, volp_dcc_hist)
portfolio_hist_adcc = make_portfolio_df(idx_adcc_hist, rp_adcc_hist, varp_adcc_hist, volp_adcc_hist)


# In[1017]:


z_hist_dcc = portfolio_hist_dcc["r_p"].values / np.clip(
    portfolio_hist_dcc["vol_p"].values,
    1e-12,
    None,
)

z_hist_adcc = portfolio_hist_adcc["r_p"].values / np.clip(
    portfolio_hist_adcc["vol_p"].values,
    1e-12,
    None,
)

var_fhs_test_dcc, es_fhs_test_dcc, hits_test_dcc = fhs_var_es(
    z_hist_init=z_hist_dcc,
    r_oos=portfolio_test_dcc["r_p"].values,
    vol_oos=portfolio_test_dcc["vol_p"].values,
    alpha=alpha,
    window=window,
)

var_fhs_test_adcc, es_fhs_test_adcc, hits_test_adcc = fhs_var_es(
    z_hist_init=z_hist_adcc,
    r_oos=portfolio_test_adcc["r_p"].values,
    vol_oos=portfolio_test_adcc["vol_p"].values,
    alpha=alpha,
    window=window,
)

print("DCC hit rate:", hits_test_dcc.mean())
print("aDCC hit rate:", hits_test_adcc.mean())


# ## Backtest
# 

# In[1018]:


results_dcc = run_backtest_suite(
    returns=portfolio_test_dcc["r_p"].values,
    var=var_fhs_test_dcc,
    es=es_fhs_test_dcc,
    alpha=alpha,
    volatility=portfolio_test_dcc["vol_p"].values,
)

results_dcc


# In[1019]:


results_adcc = run_backtest_suite(
    returns=portfolio_test_adcc["r_p"].values,
    var=var_fhs_test_adcc,
    es=es_fhs_test_adcc,
    alpha=alpha,
    volatility=portfolio_test_adcc["vol_p"].values,
)

results_adcc


# # Forecast Comparison
# 

# Forecast comparison: FZ loss plus Harvey-Leybourne-Newbold modified DM test are implemented in Python.
# 

# In[1020]:


forecast_comparison_inputs = {
    "lstm_normal": {
        "returns": portfolio_test["r_p"].values,
        "var": var_fhs_test,
        "es": es_fhs_test,
    },
    "lstm_student": {
        "returns": portfolio_test_t["r_p"].values,
        "var": var_fhs_test_t,
        "es": es_fhs_test_t,
    },
    "gru_normal": {
        "returns": portfolio_test_gru["r_p"].values,
        "var": var_fhs_test_gru,
        "es": es_fhs_test_gru,
    },
    "gru_student": {
        "returns": portfolio_test_gru_t["r_p"].values,
        "var": var_fhs_test_gru_t,
        "es": es_fhs_test_gru_t,
    },
    "bekk_lstm_scalar": {
        "returns": portfolio_test_bekk["r_p"].values,
        "var": var_fhs_test_bekk,
        "es": es_fhs_test_bekk,
    },
    "bekk_lstm_vector": {
        "returns": portfolio_test_bekk_vec["r_p"].values,
        "var": var_fhs_test_bekk_vec,
        "es": es_fhs_test_bekk_vec,
    },
    "bekk_lstm_mix": {
        "returns": portfolio_test_bekk_mix["r_p"].values,
        "var": var_fhs_test_bekk_mix,
        "es": es_fhs_test_bekk_mix,
    },
    "neural_bekk_asym_diag": {
        "returns": portfolio_test_neural_bekk["r_p"].values,
        "var": var_fhs_test_neural_bekk,
        "es": es_fhs_test_neural_bekk,
    },
    "bekk_symmetric": {
        "returns": portfolio_test_bekk_sym["r_p"].values,
        "var": var_fhs_test_bekk_sym,
        "es": es_fhs_test_bekk_sym,
    },
    "bekk_asymmetric": {
        "returns": portfolio_test_bekk_asym["r_p"].values,
        "var": var_fhs_test_bekk_asym,
        "es": es_fhs_test_bekk_asym,
    },
    "dcc": {
        "returns": portfolio_test_dcc["r_p"].values,
        "var": var_fhs_test_dcc,
        "es": es_fhs_test_dcc,
    },
    "adcc": {
        "returns": portfolio_test_adcc["r_p"].values,
        "var": var_fhs_test_adcc,
        "es": es_fhs_test_adcc,
    },
}

dm_comparisons = run_fz_dm_comparison_plan(
    model_forecasts=forecast_comparison_inputs,
    alpha=alpha,
    benchmark="bekk_symmetric",
    structured_pairs=(),
    include_pairwise=False,
)

dm_benchmark_comparisons = dm_comparisons["benchmark"]
dm_benchmark_comparisons


# In[ ]:


mcs_model_names, mcs_loss_matrix = build_fz_loss_matrix(
    model_forecasts=forecast_comparison_inputs,
    alpha=alpha,
)
mcs_fz_losses = pd.DataFrame(mcs_loss_matrix, columns=mcs_model_names)
mcs_fz_loss_path = run_dir / "fz_loss_matrix.csv"
mcs_fz_losses.to_csv(mcs_fz_loss_path, index=False)

mcs_fz_loss_summary = (
    mcs_fz_losses.mean()
    .rename("mean_fz_loss")
    .sort_values()
    .reset_index()
    .rename(columns={"index": "model"})
)
mcs_fz_loss_summary["rank"] = np.arange(1, len(mcs_fz_loss_summary) + 1)
mcs_fz_loss_summary_path = run_dir / "fz_loss_summary.csv"
mcs_fz_loss_summary.to_csv(mcs_fz_loss_summary_path, index=False)

mcs_fz_loss_summary


# # Save all Backtests

# In[1021]:


all_backtest_results = {
    "lstm_normal": results_lstm_normal,
    "lstm_student": results_lstm_student,
    "gru_normal": results_gru_normal,
    "gru_student": results_gru_student,
    "bekk_lstm_vector": results_bekk_vec,
    "bekk_lstm_scalar": results_bekk_scalar,
    "bekk_lstm_mix": results_bekk_mix,
    "neural_bekk_asym_diag": results_neural_bekk_asym_diag,
    "bekk_symmetric": results_bekk_sym,
    "bekk_asymmetric": results_bekk_asym,
    "dcc": results_dcc,
    "adcc": results_adcc,
    "forecast_comparison": dm_comparisons,
    "neural_bekk_artifacts": {
        "test_params_csv": str(neural_bekk_param_path),
        "test_param_sequences_npz": str(neural_bekk_param_npz_path),
    },
    "mcs_input": {
        "loss_function": "fissler_ziegel_patton_ziegel_chen",
        "loss_matrix_csv": str(mcs_fz_loss_path),
        "loss_summary_csv": str(mcs_fz_loss_summary_path),
        "n_obs": int(len(mcs_fz_losses)),
        "models": list(mcs_model_names),
        "mean_fz_loss": {
            row["model"]: float(row["mean_fz_loss"])
            for _, row in mcs_fz_loss_summary.iterrows()
        },
    },
}


# In[1022]:


result_save = run_dir / "all_backtest_results.json"

with open(result_save, "w", encoding="utf-8") as f:
    json.dump(all_backtest_results, f, indent=2)


# # Leptokurtosis

# In[1023]:


# Descriptive statistics: mean, std, skewness, excess kurtosis
# fuer alle Einzelserien + Equal-Weight-Portfolio

portfolio_eqw = pd.Series(
    logret[macro_tickers].to_numpy() @ w_np,
    index=logret.index,
    name="portfolio_eqw"
)

desc_stats = pd.concat(
    [
        logret[macro_tickers],
        portfolio_eqw,
    ],
    axis=1,
)

summary_table = pd.DataFrame({
    "mean": desc_stats.mean(),
    "std": desc_stats.std(),
    "skew": desc_stats.skew(),
    "excess_kurtosis": desc_stats.kurt(),
})

summary_table["raw_kurtosis"] = summary_table["excess_kurtosis"] + 3
summary_table["leptokurtic"] = summary_table["excess_kurtosis"] > 0

summary_table = summary_table.round(4)
summary_table


# $r_t \sim N(0, \Sigma_t)$ ist damit empirisch nicht haltbar.

# - ich brauche noch minimum & maximum return

# In[1024]:


thesis_table = summary_table[["mean", "std", "skew", "excess_kurtosis"]].round(4)
thesis_table


# In[1025]:


print(thesis_table.to_latex(float_format="%.4f"))


# # Notebook export als raw Code

# In[1026]:


get_ipython().system('jupyter nbconvert --to script run_models.ipynb --output run_models_code_only')

