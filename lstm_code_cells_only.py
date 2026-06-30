#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf

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

device = "mps" if torch.backends.mps.is_available() else "cpu"


# In[2]:


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


# In[3]:


#tickers = "^GSPC BTC-USD GC=F"  # S&P 500 index
macro_data = yf.download(macro_tickers, period="max")
macro_btc_data = yf.download(macro_btc_tickers, period="max")


# In[4]:


macro_data


# In[5]:


close = macro_data["Close"].copy().dropna()


# In[6]:


len(close)


# In[7]:


close.plot(title="Close Price", subplots=True, figsize=(12, 6))


# In[8]:


np.log(close).plot(title="Close Price", subplots=True, figsize=(12, 6))


# In[9]:



logret = np.log(close).diff() # log(P_t) - log(P_{t-1})) = log(P_t / P_{t-1})
logret["^TNX"] = close["^TNX"].diff() # r_t(yield) = y_t - y_{t-1}

# Aufräumen: erste Zeile NaN durch diff, ggf. weitere NaNs durch Datenlücken
logret = logret.dropna()

logret.head(), logret.tail()


# In[10]:


len(logret)


# In[11]:


import os
from pathlib import Path


# In[12]:


project_dir = Path("/Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk")
data_dir = project_dir / "data"
output_dir = project_dir / "r/output"

data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)


# In[13]:


for pattern in (
    "logret_*.csv",
    "train_df_*.csv",
    "val_df_*.csv",
    "test_df_*.csv"
):
    for old_file in data_dir.glob(pattern):
        old_file.unlink()


# In[14]:


logret


# In[15]:


logret.plot(subplots=True, figsize=(12, 6))


# In[16]:


logret.hist(bins=50, figsize=(12, 6))


# In[17]:


for col in logret.columns:
    plot_acf(logret[col], lags=40, title=f"ACF of log returns for {col}")
    plot_pacf(logret[col], lags=40, title=f"PACF of log returns for {col}")


# In[18]:


# squared returns
for col in logret.columns:
    plot_acf(logret[col]**2, lags=40, title=f"ACF of squared returns for {col}")
    plot_pacf(logret[col]**2, lags=40, title=f"PACF of squared returns for {col}")


# In[19]:


from statsmodels.tsa.stattools import adfuller

for col in logret.columns:
    print(f"ADF test for {col}:")
    result = adfuller(logret[col])
    print(f"ADF statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("---")



# In[20]:


#feature_cols = macro_tickers.split(" ")


# In[21]:


macro_tickers


# In[22]:


R = logret[macro_tickers].values.astype(np.float32)   # (N,d)

# Outer products pro t: (N,d,1) * (N,1,d) -> (N,d,d)
U = R[:, :, None] * R[:, None, :]


# In[23]:


cross_returns = vech(torch.tensor(U))
cross_returns.shape


# In[24]:


logret.shape


# In[25]:


input_data = np.concatenate([logret[macro_tickers].values, cross_returns], axis=1)
input_df = pd.DataFrame(input_data, columns=macro_tickers + [f"cross_{i+1}" for i in range(cross_returns.shape[1])])


# In[26]:


logret.index


# In[27]:


input_df.index = logret.index


# In[28]:


input_df


# In[29]:


cross_df = pd.DataFrame(cross_returns, columns=[f"cross_{i+1}" for i in range(cross_returns.shape[1])])


# In[30]:


cross_df


# In[31]:


def train_val_test_split(df, train_size=0.7, val_size=0.15):
    n = len(df)
    test_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:test_end]
    val_df = df.iloc[test_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


# In[32]:


train_df, val_df, test_df = train_val_test_split(input_df)


# In[33]:


train_df.index.name


# In[34]:


import datetime


# In[35]:


run_id = datetime.datetime.now().strftime("%Y-%m-%d")

exports = {
    "logret": logret[macro_tickers],
    "train_df": train_df[macro_tickers],
    "val_df": val_df[macro_tickers],
    "test_df": test_df[macro_tickers],
}

for name, df_export in exports.items():
    df_export = df_export.copy()
    df_export.index.name = "Date"
    df_export.to_csv(data_dir / f"{name}_{run_id}.csv")

print(f"Exportiert mit run_id: {run_id}")


# In[36]:


#feature_cols = tickers.split(" ")
mu = train_df.mean()
sigma = train_df.std() # vielleicht nicht durch die Std teilen

def normalize(df, mu, sigma, with_std=True):
    if with_std:
        return (df - mu) / sigma
    else:
        return df - mu


# In[37]:


lookback = 60
batchsize = 128


# In[38]:


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

vol_train_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=True)
vol_val_loader = DataLoader(vol_val_ds, batch_size=batchsize, shuffle=False)
vol_test_loader = DataLoader(vol_test_ds, batch_size=batchsize, shuffle=False)


# In[39]:


len(vol_train_ds), len(vol_val_ds), len(vol_test_ds)


# In[40]:


train_cov = torch.cov(torch.tensor(train_norm[macro_tickers].values, dtype=torch.float32,).T)  # sample / long-term covariance from training data
train_cov


# In[41]:


from LSTMCovariance import LSTMCovariance
from training_functions import train_covariance_model, gaussian_nll, student_nll


# In[42]:


k = len(macro_tickers)*(len(macro_tickers)+1)//2 # Anzahl an Cross Returns, die zusätzlich zu den normalen Returns als Input-Features für die Kovarianzmodellierung genutzt werden

cov_model = LSTMCovariance(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=64, num_layers=2, dropout=0.2)
cov_model, hist = train_covariance_model(
    cov_model,
    vol_train_loader,
    vol_val_loader,
    epochs=500,
    lr=1e-4,
    plateau_patience=20,
    device=device,
    scheduler_type="cosine",  # oder "cosine"
)


# In[43]:


from plots import plot_loss


# In[44]:


plot_loss(hist)


# In[45]:


sigma


# In[46]:


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


# In[47]:


vol_train_eval_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=False) # ohne Shuffle, für Plots


# In[48]:


sigma_train_scaled, y_train_scaled = predict_sigma(cov_model, vol_train_eval_loader, device=device)
sigma_train_real = scale_back(sigma_train_scaled, sigma=sigma, feature_cols=macro_tickers)


# In[49]:


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


# In[50]:


from statistics import NormalDist
from scipy.stats import t
from plots import plot_var


# In[51]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split", df=train_df, pred_vol=sigma_train_real)


# In[52]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Validation-Split", df=val_df, pred_vol=sigma_val_real)


# In[53]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Test-Split", df=test_df, pred_vol=sigma_test_real)


# In[54]:


train_df.shape, val_df.shape, test_df.shape


# In[55]:


w = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32).to(device)  # gleichgewichtetes Portfolio


# In[56]:


#w = torch.tensor([0.5, 0.4, 0.1])


# In[57]:


w.shape


# In[58]:


logret.shape


# In[59]:


torch.tensor(logret.values).shape


# In[60]:


logret_tensor = torch.tensor(logret.values, dtype=torch.float32, device=w.device)


# In[61]:


w.shape


# In[62]:


logret_tensor.shape


# In[63]:


portfolio_logrets = (w * logret_tensor).sum(dim=1)


# In[64]:


portfolio_logrets.shape


# In[65]:


portfolio_logrets_np = portfolio_logrets.detach().cpu().numpy()
plt.figure(figsize=(12, 4))
plt.plot(logret.index, portfolio_logrets_np, label="Portfolio Log-Returns")
plt.title("Portfolio Log-Returns (equal weights)")
plt.legend()
plt.tight_layout()
plt.show()


# In[66]:


#w = np.array([0.5, 0.4, 0.1], dtype=np.float32)
#w


# In[67]:


def calc_portfolio_variance(df_split, sigma_split, cols, lookback, w):
    idx = df_split.index[lookback:]
    r = df_split.loc[idx, cols].values  # (N-lookback, d)
    r_p = (w*r).sum(axis=1)  # (N-lookback,) Portfolio-Logreturns
    tmp = sigma_split @ w          # (T, d)  = Σ_t w
    var_p = (w * tmp).sum(axis=1) # (1,) = w^T (Σ_t w) Portfolio-Varianz pro t
    vol_p = np.sqrt(var_p)

    return idx, r_p, var_p, vol_p


# In[68]:


w_np = w.detach().cpu().numpy()


# In[69]:


sigma_train_real.shape


# In[70]:


idx_test, rp_test, varp_test, volp_test = calc_portfolio_variance(
    test_df, sigma_test_real, macro_tickers, lookback, w_np
)

idx_val, rp_val, varp_val, volp_val = calc_portfolio_variance(
    val_df, sigma_val_real, macro_tickers, lookback, w_np
)

idx_train, rp_train, varp_train, volp_train = calc_portfolio_variance(
    train_df, sigma_train_real, macro_tickers, lookback, w_np
)


# In[71]:


r = np.asarray(rp_test, dtype=float)
v = np.asarray(volp_test, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[72]:


def make_portfolio_df(idx, rp, varp, volp):
    return pd.DataFrame(
        {
            "r_p": np.asarray(rp, dtype=float),
            "var_p": np.asarray(varp, dtype=float),
            "vol_p": np.asarray(volp, dtype=float),
        },
        index=idx,
    )


# In[73]:


portfolio_test = make_portfolio_df(idx_test, rp_test, varp_test, volp_test)
portfolio_val = make_portfolio_df(idx_val, rp_val, varp_val, volp_val)
portfolio_train = make_portfolio_df(idx_train, rp_train, varp_train, volp_train)


# In[74]:


test_df.columns


# In[75]:


macro_tickers


# In[76]:


plot_var(alpha=0.01, lookback=lookback, view="Test-Split", portfolio=True, portfolio_df=portfolio_test)


# In[77]:


plot_var(alpha=0.01, lookback=lookback, view="Val-Split", portfolio=True, portfolio_df=portfolio_val)


# In[78]:


plot_var(alpha=0.01, lookback=lookback, view="Train-Split", portfolio=True, portfolio_df=portfolio_train)


# In[79]:


from training_functions import StudentTLoss


# In[80]:


student_loss = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[81]:


k = len(macro_tickers)*(len(macro_tickers)+1)//2 # Anzahl an Cross Returns, die zusätzlich zu den normalen Returns als Input-Features für die Kovarianzmodellierung genutzt werden

cov_model_t = LSTMCovariance(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=64, num_layers=2, dropout=0.2)
cov_model_t, hist_t = train_covariance_model(
    cov_model_t,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss,
    loss_kwargs=None,
    epochs=500,
    lr=1e-4,
    plateau_patience=20,
    device=device,
    scheduler_type="cosine",  # oder "cosine"
)


# In[82]:


learned_nu = float(student_loss.nu.detach().cpu())
student_kwargs = {"nu": learned_nu}


# In[83]:


plot_loss(hist)


# In[84]:


sigma_val_scaled_t, y_val_scale_t = predict_sigma(cov_model_t, vol_val_loader, device=device)
sigma_test_scaled_t, y_test_scaled_t = predict_sigma(cov_model_t, vol_test_loader, device=device)

sigma_val_real_t = scale_back(sigma_val_scaled_t, sigma=sigma, feature_cols=macro_tickers)
sigma_test_real_t = scale_back(sigma_test_scaled_t, sigma=sigma, feature_cols=macro_tickers)


# In[85]:


vol_train_eval_loader = DataLoader(vol_train_ds, batch_size=batchsize, shuffle=False)
sigma_train_scaled_t, y_train_scaled_t = predict_sigma(cov_model_t, vol_train_eval_loader, device=device)
sigma_train_real_t = scale_back(sigma_train_scaled_t, sigma=sigma, feature_cols=macro_tickers)


# In[86]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL)", df=train_df, pred_vol=sigma_train_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[87]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL)", df=val_df, pred_vol=sigma_val_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[88]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL)", df=test_df, pred_vol=sigma_test_real_t, loss_fn=student_nll, loss_kwargs=student_kwargs)


# In[89]:


idx_test_t, rp_test_t, varp_test_t, volp_test_t = calc_portfolio_variance(
    test_df, sigma_test_real_t, macro_tickers, lookback, w_np
)

idx_val_t, rp_val_t, varp_val_t, volp_val_t = calc_portfolio_variance(
    val_df, sigma_val_real_t, macro_tickers, lookback, w_np
)

idx_train_t, rp_train_t, varp_train_t, volp_train_t = calc_portfolio_variance(
    train_df, sigma_train_real_t, macro_tickers, lookback, w_np
)


# In[90]:


r = np.asarray(rp_test_t, dtype=float)
v = np.asarray(volp_test_t, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[91]:


portfolio_test_t = make_portfolio_df(idx_test_t, rp_test_t, varp_test_t, volp_test_t)
portfolio_val_t = make_portfolio_df(idx_val_t, rp_val_t, varp_val_t, volp_val_t)
portfolio_train_t = make_portfolio_df(idx_train_t, rp_train_t, varp_train_t, volp_train_t)


# In[92]:


from risk_metrics import fhs_var_es


# In[93]:


alpha = 0.01
window = 1000  # z.B. 250/500/1000 testen


# In[94]:


len(train_df), len(val_df), len(test_df)


# In[95]:


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


# In[96]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_test.index, portfolio_test["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_test.index, var_fhs_test, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_test.index, var_fhs_test, np.zeros_like(var_fhs_test), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Test)")
plt.legend()
plt.tight_layout()
plt.show()


# In[97]:


z_train_burn_in = z_train[:window]


# In[98]:


var_fhs_train, es_fhs_train, hit_train = fhs_var_es(
    z_hist_init=z_train_burn_in, # custom burn-in für Historie nötig
    r_oos=portfolio_train["r_p"][window:].values,
    vol_oos=portfolio_train["vol_p"][window:].values,
    alpha=alpha,
    window=window
)

hit_rate_train = hit_train.mean()
print(f"FHS Train hit rate = {hit_rate_train:.3%} (target {alpha:.3%})")


# In[99]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_train.index, portfolio_train["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_train[window:].index, var_fhs_train, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_train[window:].index, var_fhs_train, np.zeros_like(var_fhs_train), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Train)")
plt.legend()
plt.tight_layout()
plt.show()


# In[100]:


var_fhs_val, es_fhs_val, hit_val = fhs_var_es(
    z_hist_init=z_train,
    r_oos=portfolio_val["r_p"].values,
    vol_oos=portfolio_val["vol_p"].values,
    alpha=alpha,
    window=window
)

hit_rate_val = hit_val.mean()
print(f"FHS Val hit rate = {hit_rate_val:.3%} (target {alpha:.3%})")


# In[101]:


plt.figure(figsize=(12,4))
plt.plot(portfolio_val.index, portfolio_val["r_p"].values, color="black", lw=0.8, label="Portfolio Return")
plt.plot(portfolio_val.index, var_fhs_val, "--", color="tab:red", lw=1.2, label=f"FHS {alpha:.1%} VaR (lower)")
plt.fill_between(portfolio_val.index, var_fhs_val, np.zeros_like(var_fhs_val), color="tab:red", alpha=0.08)
plt.title("Filtered Historical Simulation VaR (Validation)")
plt.legend()
plt.tight_layout()
plt.show()


# In[102]:


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


# In[103]:


from bekk_kernel import BEKKCell, BEKKLSTM


# In[104]:


#loss_kwargs = {"nu": 10.0}


# In[105]:


train_cov = torch.cov(torch.tensor(train_norm[macro_tickers].values, dtype=torch.float32,).T)  # sample / long-term covariance from training data
train_cov


# In[106]:


student_loss_bekk = StudentTLoss(init_nu=8.0, min_nu=2.01, max_nu=100.0)


# In[107]:


bekk_kernel_lstm = BEKKLSTM(input_size=(len(macro_tickers)+k), n_assets=len(macro_tickers), hidden_size=64, asym=True, Sigma0=train_cov)
bekk_kernel_lstm, hist_bekk = train_covariance_model(
    bekk_kernel_lstm,
    vol_train_loader,
    vol_val_loader,
    loss_fn=student_loss_bekk,
    loss_kwargs=None,
    epochs=500,
    lr=1e-4,
    plateau_patience=20,
    device=device,
    scheduler_type="cosine",  # oder "cosine"
)


# In[108]:


learned_nu_bekk = float(student_loss_bekk.nu.detach().cpu())
bekk_student_kwargs = {"nu": learned_nu_bekk}


# In[144]:


learned_nu, learned_nu_bekk


# In[109]:


plot_loss(hist_bekk)


# In[110]:


sigma_val_scaled_bekk, y_val_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_val_loader, device=device)
sigma_val_real_bekk = scale_back(sigma_val_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)

sigma_test_scaled_bekk, y_test_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_test_loader, device=device)
sigma_test_real_bekk = scale_back(sigma_test_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)


sigma_train_scaled_bekk, y_train_scaled_bekk = predict_sigma(bekk_kernel_lstm, vol_train_eval_loader, device=device)
sigma_train_real_bekk = scale_back(sigma_train_scaled_bekk, sigma=sigma, feature_cols=macro_tickers)


# In[111]:


plot_var(alpha=0.05, lookback=lookback, cols=macro_tickers, view="Train-Split (Student-t NLL + BEKK Kernel)", df=train_df, pred_vol=sigma_train_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[112]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Val-Split (Student-t NLL + BEKK Kernel)", df=val_df, pred_vol=sigma_val_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[113]:


plot_var(alpha=0.01, lookback=lookback, cols=macro_tickers, view="Test-Split (Student-t NLL + BEKK Kernel)", df=test_df, pred_vol=sigma_test_real_bekk, loss_fn=student_nll, loss_kwargs=bekk_student_kwargs)


# In[114]:


idx_test_bekk, rp_test_bekk, varp_test_bekk, volp_test_bekk = calc_portfolio_variance(
    test_df, sigma_test_real_bekk, macro_tickers, lookback, w_np
)

idx_val_bekk, rp_val_bekk, varp_val_bekk, volp_val_bekk = calc_portfolio_variance(
    val_df, sigma_val_real_bekk, macro_tickers, lookback, w_np
)

idx_train_bekk, rp_train_bekk, varp_train_bekk, volp_train_bekk = calc_portfolio_variance(
    train_df, sigma_train_real_bekk, macro_tickers, lookback, w_np
)


# In[115]:


r = np.asarray(rp_test_bekk, dtype=float)
v = np.asarray(volp_test_bekk, dtype=float)

plt.figure(figsize=(12, 4))
plt.plot(idx_test_bekk, r, label="Portfolio Returns", color="black", lw=0.9, alpha=0.8)
plt.plot(idx_test_bekk, v, label="Predicted Vol (rescaled to return axis)", color="tab:blue", lw=1.3)
plt.title("Portfolio Returns + Predicted Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# In[116]:


portfolio_test_bekk = make_portfolio_df(idx_test_bekk, rp_test_bekk, varp_test_bekk, volp_test_bekk)
portfolio_val_bekk = make_portfolio_df(idx_val_bekk, rp_val_bekk, varp_val_bekk, volp_val_bekk)
portfolio_train_bekk = make_portfolio_df(idx_train_bekk, rp_train_bekk, varp_train_bekk, volp_train_bekk)


# In[117]:


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


# In[118]:


len(close)


# In[119]:


#importlib.reload(backtesting)


# In[120]:


from backtesting import run_backtest_suite


# In[121]:


results = run_backtest_suite(
    returns=portfolio_test["r_p"].values,
    var=var_fhs_test,
    es=es_fhs_test,
    alpha=alpha,
    volatility=portfolio_test["vol_p"].values,  # optional, aber für CC/ER sinnvoll
)

results


# In[122]:


results_student = run_backtest_suite(
    returns=portfolio_test_t["r_p"].values,
    var=var_fhs_test_t,
    es=es_fhs_test_t,
    alpha=alpha,
    volatility=portfolio_test_t["vol_p"].values,
)

results_student


# In[123]:


results_bekk = run_backtest_suite(
    returns=portfolio_test_bekk["r_p"].values,
    var=var_fhs_test_bekk,
    es=es_fhs_test_bekk,
    alpha=alpha,
    volatility=portfolio_test_bekk["vol_p"].values,
)

results_bekk


# In[124]:


import shutil
import subprocess


# In[125]:


rscript = shutil.which("Rscript")
if rscript is None:
    raise RuntimeError("Rscript wurde nicht gefunden. Prüfe, ob R im PATH liegt.")


# In[126]:


cmd = [
    rscript,
    str(project_dir / "r" / "bekks.R"),
    "--run-id", run_id,
    "--project-dir", str(project_dir)
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

print(f"BEKK run finished: {run_id}")


# In[127]:


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

sym_path = output_dir / f"bekk_forecasts_symmetric_{run_id}.csv"
asym_path = output_dir / f"bekk_forecasts_asymmetric_{run_id}.csv"

bekk_sym_df, H_bekk_sym = load_bekk_covariances(sym_path, d=len(macro_tickers))
bekk_asym_df, H_bekk_asym = load_bekk_covariances(asym_path, d=len(macro_tickers))

if len(H_bekk_sym) != len(test_df):
    raise ValueError(f"Symmetric BEKK length mismatch: {len(H_bekk_sym)} vs test_df {len(test_df)}")

if len(H_bekk_asym) != len(test_df):
    raise ValueError(f"Asymmetric BEKK length mismatch: {len(H_bekk_asym)} vs test_df {len(test_df)}")

print("Symmetric BEKK:", H_bekk_sym.shape)
print("Asymmetric BEKK:", H_bekk_asym.shape)


# In[128]:


sigma_test_real.shape


# In[129]:


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


# In[130]:


portfolio_test_bekk_sym = make_portfolio_df(idx_bekk_sym, rp_bekk_sym, varp_bekk_sym, volp_bekk_sym)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (BEKK Symmetric)", portfolio=True, portfolio_df=portfolio_test_bekk_sym)


# In[131]:


portfolio_test_bekk_asym = make_portfolio_df(idx_bekk_asym, rp_bekk_asym, varp_bekk_asym, volp_bekk_asym)

plot_var(alpha=0.01, lookback=lookback, view="Test-Split (BEKK Asymmetric)", portfolio=True, portfolio_df=portfolio_test_bekk_asym)


# In[132]:


trainval_sym_path = output_dir / f"bekk_forecasts_fitted_train_val_symmetric_{run_id}.csv"
trainval_asym_path = output_dir / f"bekk_forecasts_fitted_train_val_asymmetric_{run_id}.csv"

_, H_bekk_sym_trainval = load_bekk_covariances(trainval_sym_path, d=len(macro_tickers))
_, H_bekk_asym_trainval = load_bekk_covariances(trainval_asym_path, d=len(macro_tickers))


# In[133]:


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


# In[134]:


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


# In[135]:


results_bekk_sym = run_backtest_suite(
    returns=portfolio_test_bekk_sym["r_p"].values,
    var=var_fhs_test_bekk_sym,
    es=es_fhs_test_bekk_sym,
    alpha=alpha,
    volatility=portfolio_test_bekk_sym["vol_p"].values,
)

results_bekk_sym


# In[136]:


results_bekk_asym = run_backtest_suite(
    returns=portfolio_test_bekk_asym["r_p"].values,
    var=var_fhs_test_bekk_asym,
    es=es_fhs_test_bekk_asym,
    alpha=alpha,
    volatility=portfolio_test_bekk_asym["vol_p"].values,
)

results_bekk_asym


# In[137]:


all_backtest_results = {
    "lstm_normal": results,
    "lstm_student": results_student,
    "neural_bekk": results_bekk,
    "bekk_symmetric": results_bekk_sym,
    "bekk_asymmetric": results_bekk_asym,
}


# In[138]:


import json


# In[139]:


result_save = "results/all_backtest_results.json"

with open(project_dir / result_save, "w", encoding="utf-8") as f:
    json.dump(all_backtest_results, f, indent=2)


# In[140]:


#from rolling_forecast import run_block_reestimation


# In[141]:


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
    train_kwargs={"epochs": 200, "lr": 1e-4, "scheduler_type": "cosine"},
    verbose=False,
)
