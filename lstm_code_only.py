#!/usr/bin/env python
# coding: utf-8

# # LSTM for Covariance Matrix

# In[179]:


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

# In[180]:


# SEEDS = [42, 0, 1, 2, 3, 4]

SEED = 0

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


# In[181]:


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

# In[182]:


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


# In[183]:


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


# In[184]:


macro_data


# In[185]:


close = macro_data["Close"].copy().dropna()


# In[186]:


len(close)


# In[187]:


close.plot(title="Close Price", subplots=True, figsize=(12, 6))


# In[188]:


np.log(close).plot(title="Close Price", subplots=True, figsize=(12, 6))


# In[189]:



logret = np.log(close).diff() # log(P_t) - log(P_{t-1})) = log(P_t / P_{t-1})
logret["^TNX"] = close["^TNX"].diff() # r_t(yield) = y_t - y_{t-1}

# Aufräumen: erste Zeile NaN durch diff, ggf. weitere NaNs durch Datenlücken
logret = logret.dropna()

logret.head(), logret.tail()


# In[190]:


len(logret)


# ## Data Export

# In[191]:


import os
from pathlib import Path


# In[192]:


project_dir = Path("/Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk")
data_dir = project_dir / "data"
output_dir = project_dir / "r/output"

data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)


# In[193]:


# Versionierte Datensplits nicht loeschen.
# Dateien werden ueber DATA_ID eindeutig benannt.


# In[194]:


logret


# In[195]:


logret.plot(subplots=True, figsize=(12, 6))


# In[196]:


logret.hist(bins=50, figsize=(12, 6))


# In[197]:


for col in logret.columns:
    plot_acf(logret[col], lags=40, title=f"ACF of log returns for {col}")
    plot_pacf(logret[col], lags=40, title=f"PACF of log returns for {col}")


# In[198]:


# squared returns
for col in logret.columns:
    plot_acf(logret[col]**2, lags=40, title=f"ACF of squared returns for {col}")
    plot_pacf(logret[col]**2, lags=40, title=f"PACF of squared returns for {col}")


# In[199]:


from statsmodels.tsa.stattools import adfuller

for col in logret.columns:
    print(f"ADF test for {col}:")
    result = adfuller(logret[col])
    print(f"ADF statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("---")



# In[200]:


#feature_cols = macro_tickers.split(" ")


# In[201]:


macro_tickers


# In[202]:


R = logret[macro_tickers].values.astype(np.float32)   # (N,d)

# Outer products pro t: (N,d,1) * (N,1,d) -> (N,d,d)
U = R[:, :, None] * R[:, None, :]


# In[203]:


cross_returns = vech(torch.tensor(U))
cross_returns.shape


# In[204]:


logret.shape


# In[205]:


input_data = np.concatenate([logret[macro_tickers].values, cross_returns], axis=1)
input_df = pd.DataFrame(input_data, columns=macro_tickers + [f"cross_{i+1}" for i in range(cross_returns.shape[1])])


# In[206]:


logret.index


# In[207]:


input_df.index = logret.index


# In[208]:


input_df


# In[209]:


cross_df = pd.DataFrame(cross_returns, columns=[f"cross_{i+1}" for i in range(cross_returns.shape[1])])


# In[210]:


cross_df


# In[211]:


def train_val_test_split(df, train_size=0.7, val_size=0.15):
    n = len(df)
    test_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:test_end]
    val_df = df.iloc[test_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


# In[212]:


train_df, val_df, test_df = train_val_test_split(input_df)


# In[213]:


train_df.index.name


# In[214]:


import datetime


# In[215]:


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

# In[216]:


#feature_cols = tickers.split(" ")
mu = train_df.mean()
sigma = train_df.std() # vielleicht nicht durch die Std teilen

def normalize(df, mu, sigma, with_std=True):
    if with_std:
        return (df - mu) / sigma
    else:
        return df - mu


# ## Hyper Paramters

# In[217]:


lookback = 40
batchsize = 64
hidden_size = 64
dropout = 0.1
lr = 3e-4
bekk_lr = 1e-4
bekk_mix_lr = 5e-5
bekk_jitter = 1e-4
bekk_device = "cpu"
early_stopping_patience = 50
early_stopping_min_delta = 1e-4


# ## Data Loader

# In[218]:


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# In[219]:


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


# In[220]:


len(vol_train_ds), len(vol_val_ds), len(vol_test_ds)


# ```train_cov```ggf. zur Initialisierung nutzen. 

# In[221]:


train_cov = torch.cov(torch.tensor(train_norm[macro_tickers].values, dtype=torch.float32,).T)  # sample / long-term covariance from training data
train_cov


# In[222]:


from LSTMCovariance import LSTMCovariance
from training_functions import train_covariance_model, gaussian_nll, student_nll


# In[223]:


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


# In[224]:


from plots import plot_loss


# In[225]:


plot_loss(hist)


# In[226]:


sigma


# In[227]:


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


# In[228]:


vol_train_eval_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=False) # ohne Shuffle, für Plots


# In[229]:


sigma_train_scaled, y_train_scaled = predict_sigma(cov_model, vol_train_eval_loader, device=device)
sigma_train_real = scale_back(sigma_train_scaled, sigma=sigma, feature_cols=macro_tickers)


# In[230]:


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


# In[231]:


from statistics import NormalDist
from scipy.stats import t
from plots import plot_var


# In[232]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split", df=train_df, pred_vol=sigma_train_real)


# In[233]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Validation-Split", df=val_df, pred_vol=sigma_val_real)


# In[234]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Test-Split", df=test_df, pred_vol=sigma_test_real)


# In[235]:


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

# In[236]:


w = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32).to(device)  # gleichgewichtetes Portfolio


# In[237]:


#w = torch.tensor([0.5, 0.4, 0.1])


# In[238]:


w.shape


# In[239]:


logret.shape


# In[240]:


torch.tensor(logret.values).shape


# In[241]:


logret_tensor = torch.tensor(logret.values, dtype=torch.float32, device=w.device)


# In[242]:


w.shape


# In[243]:


logret_tensor.shape


# In[244]:


portfolio_logrets = (w * logret_tensor).sum(dim=1)


# In[245]:


portfolio_logrets.shape


# In[246]:


portfolio_logrets_np = portfolio_logrets.detach().cpu().numpy()
plt.figure(figsize=(12, 4))
plt.plot(logret.index, portfolio_logrets_np, label="Portfolio Log-Returns")
plt.title("Portfolio Log-Returns (equal weights)")
plt.legend()
plt.tight_layout()
plt.show()


# In[247]:


#w = np.array([0.5, 0.4, 0.1], dtype=np.float32)
#w


# In[248]:


def calc_portfolio_variance(df_split, sigma_split, cols, lookback, w):
    idx = df_split.index[lookback:]
    r = df_split.loc[idx, cols].values  # (N-lookback, d)
    r_p = (w*r).sum(axis=1)  # (N-lookback,) Portfolio-Logreturns
    tmp = sigma_split @ w          # (T, d)  = Σ_t w
    var_p = (w * tmp).sum(axis=1) # (1,) = w^T (Σ_t w) Portfolio-Varianz pro t
    vol_p = np.sqrt(var_p)

    return idx, r_p, var_p, vol_p


# ## LSTM mit Normal Loss

# In[249]:


w_np = w.detach().cpu().numpy()


# In[250]:


sigma_train_real.shape


# In[251]:


idx_test, rp_test, varp_test, volp_test = calc_portfolio_variance(
    test_df, sigma_test_real, macro_tickers, lookback, w_np
)

idx_val, rp_val, varp_val, volp_val = calc_portfolio_variance(
    val_df, sigma_val_real, macro_tickers, lookback, w_np
)

idx_train, rp_train, varp_train, volp_train = calc_portfolio_variance(
    train_df, sigma_train_real, macro_tickers, lookback, w_np
)


# In[252]:


r = np.asarray(rp_test, dtype=float)
v = np.asarray(volp_test, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[253]:


def make_portfolio_df(idx, rp, varp, volp):
    return pd.DataFrame(
        {
            "r_p": np.asarray(rp, dtype=float),
            "var_p": np.asarray(varp, dtype=float),
            "vol_p": np.asarray(volp, dtype=float),
        },
        index=idx,
    )


# In[254]:


portfolio_test = make_portfolio_df(idx_test, rp_test, varp_test, volp_test)
portfolio_val = make_portfolio_df(idx_val, rp_val, varp_val, volp_val)
portfolio_train = make_portfolio_df(idx_train, rp_train, varp_train, volp_train)


# In[255]:


test_df.columns


# In[256]:


macro_tickers


# In[257]:


plot_var(alpha=0.01, lookback=lookback, view="Test-Split", portfolio=True, portfolio_df=portfolio_test)


# In[258]:


plot_var(alpha=0.01, lookback=lookback, view="Val-Split", portfolio=True, portfolio_df=portfolio_val)


# In[259]:


plot_var(alpha=0.01, lookback=lookback, view="Train-Split", portfolio=True, portfolio_df=portfolio_train)


# ## LSTM mit Student-t Loss

# ### Training

# Modell mit Studen-t-Verteilung

# $$r_t \sim t_\nu(0, S_t)$$
# 
# $$ S_t = \frac{\nu}{\nu-2} \Sigma_t = \frac{\nu}{\nu-2} L_t L_t^\top = (\sqrt{\frac{\nu}{\nu-2}} L) \cdot (\sqrt{\frac{\nu}{\nu-2}} L)^\top$$

# In[260]:


from training_functions import StudentTLoss


# In[261]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[262]:


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


# In[263]:


learned_nu = float(student_loss.nu.detach().cpu())
student_kwargs = {"nu": learned_nu}


# In[264]:


plot_loss(hist)


# In[265]:


sigma_val_scaled_t, y_val_scale_t = predict_sigma(cov_model_t, vol_val_loader, device=device)
sigma_test_scaled_t, y_test_scaled_t = predict_sigma(cov_model_t, vol_test_loader, device=device)

sigma_val_real_t = scale_back(sigma_val_scaled_t, sigma=sigma, feature_cols=macro_tickers)
sigma_test_real_t = scale_back(sigma_test_scaled_t, sigma=sigma, feature_cols=macro_tickers)


# In[266]:


vol_train_eval_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=False)
sigma_train_scaled_t, y_train_scaled_t = predict_sigma(cov_model_t, vol_train_eval_loader, device=device)
sigma_train_real_t = scale_back(sigma_train_scaled_t, sigma=sigma, feature_cols=macro_tickers)


# In[267]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL)", df=train_df, pred_vol=sigma_train_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[268]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL)", df=val_df, pred_vol=sigma_val_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[269]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL)", df=test_df, pred_vol=sigma_test_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# ### Portfolio Returns

# - Hier müssten jetzt noch mal die Portfolio Returns & Volas mit ```make_portfolio_df``` berechnet werden für Student-t Verteilung!

# In[270]:


idx_test_t, rp_test_t, varp_test_t, volp_test_t = calc_portfolio_variance(
    test_df, sigma_test_real_t, macro_tickers, lookback, w_np
)

idx_val_t, rp_val_t, varp_val_t, volp_val_t = calc_portfolio_variance(
    val_df, sigma_val_real_t, macro_tickers, lookback, w_np
)

idx_train_t, rp_train_t, varp_train_t, volp_train_t = calc_portfolio_variance(
    train_df, sigma_train_real_t, macro_tickers, lookback, w_np
)


# In[271]:


r = np.asarray(rp_test_t, dtype=float)
v = np.asarray(volp_test_t, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[272]:


portfolio_test_t = make_portfolio_df(idx_test_t, rp_test_t, varp_test_t, volp_test_t)
portfolio_val_t = make_portfolio_df(idx_val_t, rp_val_t, varp_val_t, volp_val_t)
portfolio_train_t = make_portfolio_df(idx_train_t, rp_train_t, varp_train_t, volp_train_t)


# Modell mit skewed Studen-t-Verteilung

# ## Filtered Historical Simulation

# - bisher FHS nur mit Normal Volas

# In[273]:


from risk_metrics import fhs_var_es


# In[274]:


alpha = 0.01
window = 1000  # z.B. 250/500/1000 testen


# In[275]:


len(train_df), len(val_df), len(test_df)


# ### Normal Loss (Portfolio)

# In[276]:


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


# In[277]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_test.index, portfolio_test["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_test.index, var_fhs_test, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_test.index, var_fhs_test, np.zeros_like(var_fhs_test), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Test)")
plt.legend()
plt.tight_layout()
plt.show()


# - Falls FHS auf train genutzt werden soll, muss zuvor ein z_train_burn_in genutzt werden wo zb die ersten 500 z_t aus train genutzt werden für die historie und von dort aus wird dann der VaR rollierend berechnet!

# In[278]:


z_train_burn_in = z_train[:window]


# In[279]:


var_fhs_train, es_fhs_train, hit_train = fhs_var_es(
    z_hist_init=z_train_burn_in, # custom burn-in für Historie nötig
    r_oos=portfolio_train["r_p"][window:].values,
    vol_oos=portfolio_train["vol_p"][window:].values,
    alpha=alpha,
    window=window
)

hit_rate_train = hit_train.mean()
print(f"FHS Train hit rate = {hit_rate_train:.3%} (target {alpha:.3%})")


# In[280]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_train.index, portfolio_train["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_train[window:].index, var_fhs_train, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_train[window:].index, var_fhs_train, np.zeros_like(var_fhs_train), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Train)")
plt.legend()
plt.tight_layout()
plt.show()


# In[281]:


var_fhs_val, es_fhs_val, hit_val = fhs_var_es(
    z_hist_init=z_train,
    r_oos=portfolio_val["r_p"].values,
    vol_oos=portfolio_val["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_val = hit_val.mean()
print(f"FHS Val hit rate = {hit_rate_val:.3%} (target {alpha:.3%})")


# In[282]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_val.index, portfolio_val["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_val.index, var_fhs_val, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_val.index, var_fhs_val, np.zeros_like(var_fhs_val), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Validation)")
plt.legend()
plt.tight_layout()
plt.show()


# ### Student-t Loss (Portfolio)

# In[283]:


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
# $x_{t-1} := [
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

# In[284]:


from bekk_kernel import BEKKCell, BEKKLSTM


# - Es müssen neue Dataset & Dataloader angelegt werden, da die MGARCH Kernel Function ein BEKK(1,1) ist. Daher ist der bisherige ```lookback=60``` nicht passend.

# In[285]:


#loss_kwargs = {"nu": 10.0}


# ```train_cov```ggf. zur Initialisierung nutzen. 

# In[286]:


train_cov = torch.cov(torch.tensor(train_norm[macro_tickers].values, dtype=torch.float32,).T)  # sample / long-term covariance from training data
train_cov


# In[287]:


Sigma0 = torch.tensor(np.cov((train_df[macro_tickers].values * 100).T),
                      dtype=torch.float32)


# In[357]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_bekk = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# ## Scalar Modulation

# In[ ]:


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
    model_name="neural_bekk_scalar",
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


# In[290]:


learned_nu_bekk = float(student_loss_bekk.nu.detach().cpu())
bekk_student_kwargs = {"nu": learned_nu_bekk}


# In[291]:


learned_nu, learned_nu_bekk


# In[292]:


plot_loss(hist_bekk)


# In[293]:


sigma_val_scaled_bekk, y_val_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_val_loader, device=bekk_device)
sigma_val_real_bekk = scale_back(sigma_val_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk, y_test_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_test_loader, device=bekk_device)
sigma_test_real_bekk = scale_back(sigma_test_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk, y_train_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_train_eval_loader, device=bekk_device)
sigma_train_real_bekk = scale_back(sigma_train_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)


# In[294]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel)", df=train_df, pred_vol=sigma_train_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[295]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel)", df=val_df, pred_vol=sigma_val_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[296]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel)", df=test_df, pred_vol=sigma_test_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# ### Portfolio

# In[297]:


idx_test_bekk, rp_test_bekk, varp_test_bekk, volp_test_bekk = calc_portfolio_variance(
    test_df, sigma_test_real_bekk, macro_tickers, lookback, w_np
)

idx_val_bekk, rp_val_bekk, varp_val_bekk, volp_val_bekk = calc_portfolio_variance(
    val_df, sigma_val_real_bekk, macro_tickers, lookback, w_np
)

idx_train_bekk, rp_train_bekk, varp_train_bekk, volp_train_bekk = calc_portfolio_variance(
    train_df, sigma_train_real_bekk, macro_tickers, lookback, w_np
)


# In[298]:


r = np.asarray(rp_test_bekk, dtype=float)
v = np.asarray(volp_test_bekk, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[299]:


portfolio_test_bekk = make_portfolio_df(idx_test_bekk, rp_test_bekk, varp_test_bekk, volp_test_bekk)
portfolio_val_bekk = make_portfolio_df(idx_val_bekk, rp_val_bekk, varp_val_bekk, volp_val_bekk)
portfolio_train_bekk = make_portfolio_df(idx_train_bekk, rp_train_bekk, varp_train_bekk, volp_train_bekk)


# ### FHS

# In[300]:


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


# In[301]:


len(close)


# # Backtesting

# !!! Backtestings müssen mit dem selben $\alpha$ vorgenommen werden wie mit FHS !!!

# Backtests sind für univariate Zeitreihen definiert, d.h. wir testen alle Modelle im Kontext eines vordefinierten Portfolios.

# In[302]:


#importlib.reload(backtesting)


# In[303]:


from backtesting import run_backtest_suite


# In[304]:


results = run_backtest_suite(
    returns=portfolio_test["r_p"].values,
    var=var_fhs_test,
    es=es_fhs_test,
    alpha=alpha,
    volatility=portfolio_test["vol_p"].values,  # optional, aber für CC/ER sinnvoll
)

results


# In[305]:


results_student = run_backtest_suite(
    returns=portfolio_test_t["r_p"].values,
    var=var_fhs_test_t,
    es=es_fhs_test_t,
    alpha=alpha,
    volatility=portfolio_test_t["vol_p"].values,
)

results_student


# In[306]:


results_bekk = run_backtest_suite(
    returns=portfolio_test_bekk["r_p"].values,
    var=var_fhs_test_bekk,
    es=es_fhs_test_bekk,
    alpha=alpha,
    volatility=portfolio_test_bekk["vol_p"].values,
)

results_bekk


# ## Vector Modulation

# In[307]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_bekk_vec = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[308]:


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
    model_name="neural_bekk_vector",
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


# In[309]:


learned_nu_bekk_vec = float(student_loss_bekk_vec.nu.detach().cpu())
bekk_vec_student_kwargs = {"nu": learned_nu_bekk_vec}


# In[310]:


learned_nu_bekk_vec, learned_nu_bekk_vec


# In[311]:


plot_loss(hist_bekk_vec)


# In[312]:


sigma_val_scaled_bekk_vec, y_val_scaled_bekk_vec = predict_sigma(bekk_kernel_lstm_vec, vol_val_loader, device=bekk_device)
sigma_val_real_bekk_vec = scale_back(sigma_val_scaled_bekk_vec, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk_vec, y_test_scaled_bekk_vec = predict_sigma(bekk_kernel_lstm_vec, vol_test_loader, device=bekk_device)
sigma_test_real_bekk_vec = scale_back(sigma_test_scaled_bekk_vec, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk_vec, y_train_scaled_bekk_vec = predict_sigma(bekk_kernel_lstm_vec, vol_train_eval_loader, device=bekk_device)
sigma_train_real_bekk_vec = scale_back(sigma_train_scaled_bekk_vec, sigma=sigma, feature_cols=macro_tickers)


# In[313]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel)", df=train_df, pred_vol=sigma_train_real_bekk_vec, loss_fn=student_nll, loss_kwargs=bekk_vec_student_kwargs)


# In[314]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel)", df=val_df, pred_vol=sigma_val_real_bekk_vec, loss_fn=student_nll, loss_kwargs=bekk_vec_student_kwargs)


# In[315]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel)", df=test_df, pred_vol=sigma_test_real_bekk_vec, loss_fn=student_nll, loss_kwargs=bekk_vec_student_kwargs)


# ### Portfolio

# In[316]:


idx_test_bekk_vec, rp_test_bekk_vec, varp_test_bekk_vec, volp_test_bekk_vec = calc_portfolio_variance(
    test_df, sigma_test_real_bekk_vec, macro_tickers, lookback, w_np
)

idx_val_bekk_vec, rp_val_bekk_vec, varp_val_bekk_vec, volp_val_bekk_vec = calc_portfolio_variance(
    val_df, sigma_val_real_bekk_vec, macro_tickers, lookback, w_np
)

idx_train_bekk_vec, rp_train_bekk_vec, varp_train_bekk_vec, volp_train_bekk_vec = calc_portfolio_variance(
    train_df, sigma_train_real_bekk_vec, macro_tickers, lookback, w_np
)


# In[317]:


r = np.asarray(rp_test_bekk_vec, dtype=float)
v = np.asarray(volp_test_bekk_vec, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk_vec, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk_vec, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[318]:


portfolio_test_bekk_vec = make_portfolio_df(idx_test_bekk_vec, rp_test_bekk_vec, varp_test_bekk_vec, volp_test_bekk_vec)
portfolio_val_bekk_vec = make_portfolio_df(idx_val_bekk_vec, rp_val_bekk_vec, varp_val_bekk_vec, volp_val_bekk_vec)
portfolio_train_bekk_vec = make_portfolio_df(idx_train_bekk_vec, rp_train_bekk_vec, varp_train_bekk_vec, volp_train_bekk_vec)


# ### FHS

# In[319]:


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


# In[320]:


len(close)


# ### Backtesting

# !!! Backtestings müssen mit dem selben $\alpha$ vorgenommen werden wie mit FHS !!!

# Backtests sind für univariate Zeitreihen definiert, d.h. wir testen alle Modelle im Kontext eines vordefinierten Portfolios.

# In[321]:


results_bekk_vec = run_backtest_suite(
    returns=portfolio_test_bekk_vec["r_p"].values,
    var=var_fhs_test_bekk_vec,
    es=es_fhs_test_bekk_vec,
    alpha=alpha,
    volatility=portfolio_test_bekk_vec["vol_p"].values,
)

results_bekk_vec


# ## Convex Mixture Modulation

# In[322]:


seed_everything(SEED)
vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True, generator=make_generator(SEED))
student_loss_bekk_mix = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[323]:


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
    model_name="bekk_mix",
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


# In[324]:


learned_nu_bekk_mix = float(student_loss_bekk_mix.nu.detach().cpu())
bekk_mix_student_kwargs = {"nu": learned_nu_bekk_mix}


# In[325]:


learned_nu_bekk_vec, learned_nu_bekk_mix


# In[326]:


plot_loss(hist_bekk_mix)


# In[327]:


sigma_val_scaled_bekk_mix, y_val_scaled_bekk_mix = predict_sigma(bekk_kernel_lstm_mix, vol_val_loader, device=bekk_device)
sigma_val_real_bekk_mix = scale_back(sigma_val_scaled_bekk_mix, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk_mix, y_test_scaled_bekk_mix = predict_sigma(bekk_kernel_lstm_mix, vol_test_loader, device=bekk_device)
sigma_test_real_bekk_mix = scale_back(sigma_test_scaled_bekk_mix, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk_mix, y_train_scaled_bekk_mix = predict_sigma(bekk_kernel_lstm_mix, vol_train_eval_loader, device=bekk_device)
sigma_train_real_bekk_mix = scale_back(sigma_train_scaled_bekk_mix, sigma=sigma, feature_cols=macro_tickers)


# In[328]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel Convex Mixture)", df=train_df, pred_vol=sigma_train_real_bekk_mix, loss_fn=student_nll, loss_kwargs=bekk_mix_student_kwargs)


# In[329]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel Convex Mixture)", df=val_df, pred_vol=sigma_val_real_bekk_mix, loss_fn=student_nll, loss_kwargs=bekk_mix_student_kwargs)


# In[330]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel Convex Mixture)", df=test_df, pred_vol=sigma_test_real_bekk_mix, loss_fn=student_nll, loss_kwargs=bekk_mix_student_kwargs)


# ### Portfolio

# In[331]:


idx_test_bekk_mix, rp_test_bekk_mix, varp_test_bekk_mix, volp_test_bekk_mix = calc_portfolio_variance(
    test_df, sigma_test_real_bekk_mix, macro_tickers, lookback, w_np
)

idx_val_bekk_mix, rp_val_bekk_mix, varp_val_bekk_mix, volp_val_bekk_mix = calc_portfolio_variance(
    val_df, sigma_val_real_bekk_mix, macro_tickers, lookback, w_np
)

idx_train_bekk_mix, rp_train_bekk_mix, varp_train_bekk_mix, volp_train_bekk_mix = calc_portfolio_variance(
    train_df, sigma_train_real_bekk_mix, macro_tickers, lookback, w_np
)


# In[332]:


r = np.asarray(rp_test_bekk_mix, dtype=float)
v = np.asarray(volp_test_bekk_mix, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk_mix, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk_mix, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[333]:


portfolio_test_bekk_mix = make_portfolio_df(idx_test_bekk_mix, rp_test_bekk_mix, varp_test_bekk_mix, volp_test_bekk_mix)
portfolio_val_bekk_mix = make_portfolio_df(idx_val_bekk_mix, rp_val_bekk_mix, varp_val_bekk_mix, volp_val_bekk_mix)
portfolio_train_bekk_mix = make_portfolio_df(idx_train_bekk_mix, rp_train_bekk_mix, varp_train_bekk_mix, volp_train_bekk_mix)


# ### FHS

# In[334]:


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

# In[335]:


results_bekk_mix = run_backtest_suite(
    returns=portfolio_test_bekk_mix["r_p"].values,
    var=var_fhs_test_bekk_mix,
    es=es_fhs_test_bekk_mix,
    alpha=alpha,
    volatility=portfolio_test_bekk_mix["vol_p"].values,
)

results_bekk_mix


# # BEKK(1, 1) symmetrisch & asymmetrisch
# 
# importiert aus R bzw BEKKs Library.

# ## Daten import

# In[336]:


import shutil
import subprocess


# In[337]:


rscript = shutil.which("Rscript")
if rscript is None:
    raise RuntimeError("Rscript wurde nicht gefunden. Prüfe, ob R im PATH liegt.")


# In[338]:


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


# In[339]:


def load_bekk_covariances(path, d):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"BEKK forecast file not found: {path}")

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

sym_path = output_dir / f"bekk_forecasts_symmetric_{RUN_ID}.csv"
asym_path = output_dir / f"bekk_forecasts_asymmetric_{RUN_ID}.csv"

bekk_sym_df, H_bekk_sym = load_bekk_covariances(sym_path, d=len(macro_tickers))
bekk_asym_df, H_bekk_asym = load_bekk_covariances(asym_path, d=len(macro_tickers))

if len(H_bekk_sym) != len(test_df):
    raise ValueError(f"Symmetric BEKK length mismatch: {len(H_bekk_sym)} vs test_df {len(test_df)}")

if len(H_bekk_asym) != len(test_df):
    raise ValueError(f"Asymmetric BEKK length mismatch: {len(H_bekk_asym)} vs test_df {len(test_df)}")

print("Symmetric BEKK:", H_bekk_sym.shape)
print("Asymmetric BEKK:", H_bekk_asym.shape)


# In[340]:


sigma_test_real.shape


# BEKK(1, 1) hat ```lookback``` mehr OOS predictions als die LSTM Modelle

# In[341]:


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


# In[342]:


portfolio_test_bekk_sym = make_portfolio_df(idx_bekk_sym, rp_bekk_sym, varp_bekk_sym, volp_bekk_sym)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (BEKK Symmetric)", portfolio=True, portfolio_df=portfolio_test_bekk_sym)


# In[343]:


portfolio_test_bekk_asym = make_portfolio_df(idx_bekk_asym, rp_bekk_asym, varp_bekk_asym, volp_bekk_asym)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (BEKK Asymmetric)", portfolio=True, portfolio_df=portfolio_test_bekk_asym)


# ## FHS

# In[344]:


trainval_sym_path = output_dir / f"bekk_forecasts_fitted_train_val_symmetric_{RUN_ID}.csv"
trainval_asym_path = output_dir / f"bekk_forecasts_fitted_train_val_asymmetric_{RUN_ID}.csv"

_, H_bekk_sym_trainval = load_bekk_covariances(trainval_sym_path, d=len(macro_tickers))
_, H_bekk_asym_trainval = load_bekk_covariances(trainval_asym_path, d=len(macro_tickers))


# In[345]:


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


# In[346]:


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

# In[347]:


results_bekk_sym = run_backtest_suite(
    returns=portfolio_test_bekk_sym["r_p"].values,
    var=var_fhs_test_bekk_sym,
    es=es_fhs_test_bekk_sym,
    alpha=alpha,
    volatility=portfolio_test_bekk_sym["vol_p"].values,
)

results_bekk_sym


# In[348]:


results_bekk_asym = run_backtest_suite(
    returns=portfolio_test_bekk_asym["r_p"].values,
    var=var_fhs_test_bekk_asym,
    es=es_fhs_test_bekk_asym,
    alpha=alpha,
    volatility=portfolio_test_bekk_asym["vol_p"].values,
)

results_bekk_asym


# # Save all Backtests

# In[349]:


all_backtest_results = {
    "lstm_normal": results,
    "lstm_student": results_student,
    "neural_bekk_scalar": results_bekk,
    "neural_bekk_vector": results_bekk_vec,
    "bekk_mix": results_bekk_mix,
    "bekk_symmetric": results_bekk_sym,
    "bekk_asymmetric": results_bekk_asym,
}


# In[350]:


result_save = run_dir / "all_backtest_results.json"

with open(result_save, "w", encoding="utf-8") as f:
    json.dump(all_backtest_results, f, indent=2)


# # Leptokurtosis

# In[351]:


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

# In[352]:


thesis_table = summary_table[["mean", "std", "skew", "excess_kurtosis"]].round(4)
thesis_table


# In[353]:


print(thesis_table.to_latex(float_format="%.4f"))


# # Notebook export als raw Code

# In[354]:


get_ipython().system('jupyter nbconvert --to script lstm.ipynb --output lstm_code_only')


# # Rolling Block Re-Estimation

# In[355]:


#from rolling_forecast import run_block_reestimation


# In[356]:


result = run_block_reestimation(
    data=input_df,
    target_cols=macro_tickers,
    model_factory=lambda: LSTMCovariance(
        input_size=len(input_df.columns),
        n_assets=len(macro_tickers),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ),
    lookback=60,
    train_window=1000,
    refit_every=100,
    val_size=100,
    batch_size=64,
    device=device,
    loss_fn=student_nll,
    loss_kwargs=loss_kwargs,
    train_kwargs={
        "epochs": 200,
        "lr": 1e-4,
        "scheduler_type": "cosine",
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    },
    verbose=False,
)
