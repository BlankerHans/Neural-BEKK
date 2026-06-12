from statistics import NormalDist
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt
from training_functions import gaussian_nll, student_nll


def _to_float(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def plot_var(alpha=0.01, lookback=None, cols=None, view="Test-Split", df=None, pred_vol=None, portfolio=False, portfolio_df=None, loss_fn=gaussian_nll, loss_kwargs=None):
    if loss_kwargs is None:
        loss_kwargs = {}

    is_student = loss_fn == student_nll or hasattr(loss_fn, "nu")

    if loss_fn is None or loss_fn == gaussian_nll:
        z_lo = NormalDist().inv_cdf(alpha)        # z_{alpha}, z.B. -2.326
        z_hi = NormalDist().inv_cdf(1.0 - alpha)  # z_{1-alpha}, z.B. +2.326
        nu = None
    elif is_student:
        if hasattr(loss_fn, "nu"):
            nu = _to_float(loss_fn.nu)
        else:
            if "nu" not in loss_kwargs:
                raise ValueError("Bei Student-t bitte loss_kwargs={'nu': learned_nu} übergeben.")
            nu = _to_float(loss_kwargs["nu"])
        z_lo = t.ppf(alpha, df=nu)
        z_hi = t.ppf(1-alpha, df=nu)
    else:
        raise ValueError("Unbekannte loss_fn. Bitte gaussian_nll, student_nll oder StudentTLoss übergeben.")
    
    if portfolio:
        if portfolio_df is None:
            raise ValueError("Bei portfolio=True bitte portfolio_df übergeben.")
        
        idx = portfolio_df.index
        r = portfolio_df["r_p"].values
        vol = portfolio_df["vol_p"].values

        if is_student:
            vol = vol * np.sqrt((nu - 2.0) / nu)

        var_lo = z_lo * vol
        var_hi = z_hi * vol

        plt.figure(figsize=(12, 4))
        plt.plot(idx, r, color="black", lw=0.8, label="Portfolio Return", alpha=0.8) 
        plt.plot(idx, var_lo, "--", color="tab:red", lw=1.2, label=f"{alpha:.1%} VaR (lower)")
        plt.plot(idx, var_hi, "--", color="tab:green", lw=1.2, label=f"{1-alpha:.1%} Quantile (upper)")
        plt.fill_between(idx, var_lo, var_hi, alpha=0.12, color="gray")
        plt.title(f"Portfolio: VaR-Bänder ({view})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        hit_rate = np.mean(r <= var_lo)
        print(f"Portfolio hit rate = {hit_rate:.3%} (target {alpha:.3%})")
        return

    if df is None:
        raise ValueError("Bei portfolio=False bitte df übergeben.")
    if cols is None:
        raise ValueError("Bei portfolio=False bitte cols übergeben.")
    if pred_vol is None:
        raise ValueError("Bei portfolio=False bitte pred_vol übergeben.")
    if lookback is None:
        raise ValueError("Bitte lookback übergeben.")

    index_vol = df.index[lookback:]
    rets = df.loc[index_vol, cols].copy()

    for i, col in enumerate(cols):
        # sigma_{t+1} aus deiner vorhergesagten Kovarianz Matrix für die Asset i extrahieren
        if is_student:
            sigma_t = np.sqrt(np.clip(pred_vol[:, i, i] * (nu - 2.0) / nu, 1e-12, None)) # scale matrix nicht kovarianzmatrix
        else:
            sigma_t = np.sqrt(np.clip(pred_vol[:, i, i], 1e-12, None))


        var_lo = z_lo * sigma_t   # downside VaR-Linie
        var_hi = z_hi * sigma_t   # obere symmetrische Linie

        r = rets[col].values
        idx = rets.index

        plt.style.use("ggplot")
        plt.figure(figsize=(12, 4))
        plt.plot(idx, r, color="black", lw=0.8, label="Return", alpha=0.7)
        plt.plot(idx, var_lo, "--", color="tab:red", lw=1.2, label=f"{alpha:.1%} VaR (lower)")
        plt.plot(idx, var_hi, "--", color="tab:green", lw=1.2, label=f"{1-alpha:.1%} quantile (upper)")
        plt.fill_between(idx, var_lo, var_hi, alpha=0.12, color="gray")
        plt.title(f"{col}: VaR-Bänder ({view})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Backtest: Anteil der Unterschreitungen sollte ~ alpha sein
        hit_rate = np.mean(r <= var_lo)
        print(f"{col}: hit rate = {hit_rate:.3%} (target {alpha:.3%})")


def plot_loss(history):
    epochs = np.arange(1, len(history["train_epoch_nll"]) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    l1, = ax1.plot(epochs, history["train_epoch_nll"], color="tab:red",  lw=1.8, label="Train NLL (epoch)")
    l2, = ax1.plot(epochs, history["val_epoch_nll"],   color="tab:blue", lw=1.8, label="Val NLL (epoch)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL")
    ax1.set_title("Epoch NLL")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    l3, = ax2.plot(epochs, history["lr"], "--", color="black", lw=1.6, label="LR")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")

    handles = [l1, l2, l3]
    labels = [h.get_label() for h in handles]
    ax1.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        borderaxespad=0.6,
        handlelength=2.0,
        frameon=True,
    )

    fig.subplots_adjust(right=0.94)
    plt.show()
