from statistics import NormalDist
from scipy.stats import t
import numpy as np
import pandas as pd
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


def plot_fhs_var(
    dates,
    r,
    var,
    es=None,
    hits=None,
    alpha=0.01,
    title=None,
    crisis_windows=None,
    var_parametric=None,
    figsize=(12, 4),
    save_path=None,
    show=True,
):
    """Plot one-step-ahead FHS VaR (and ES) against realized portfolio returns.

    Parameters
    ----------
    dates : array-like
        x-axis (datetime index or range).
    r : array-like
        Realized portfolio returns.
    var : array-like
        FHS VaR series (the lower-tail line; sign as returned by ``fhs_var_es``,
        i.e. negative numbers).
    es : array-like, optional
        FHS Expected Shortfall series (plotted as a second, lower line).
    hits : array-like of bool, optional
        Violation indicators (``r <= var``); violation points are marked. If
        ``None`` they are recomputed from ``r`` and ``var``.
    alpha : float
        Tail level, only used for labels (e.g. 0.01 -> "1% VaR").
    crisis_windows : list of (start, end[, label]), optional
        Shaded background episodes; dates parseable by the x-axis.
    var_parametric : array-like, optional
        A second VaR line (e.g. Gaussian) drawn faintly for contrast.
    save_path : str or Path, optional
        If given, the figure is written there (PDF/PNG by extension).
    """
    # Reuse the thesis style and the exact crisis-shading used in the
    # section-3 figures (same windows, same grey, same labels), so this plot
    # is visually consistent with figures/section3/log_returns.pdf.
    _shade = None
    _default_windows = None
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
        from section3_eda import (
            set_thesis_style,
            _shade_crisis_windows as _shade,
            DEFAULT_CRISIS_WINDOWS as _default_windows,
        )
        set_thesis_style()
    except Exception:
        pass

    dates = np.asarray(dates)
    r = np.asarray(r, dtype=float)
    var = np.asarray(var, dtype=float)
    if hits is None:
        hits = r <= var
    hits = np.asarray(hits, dtype=bool)

    # "default" -> use the canonical thesis crisis windows.
    if crisis_windows == "default":
        crisis_windows = _default_windows

    fig, ax = plt.subplots(figsize=figsize)

    data_start = pd.Timestamp(dates[0]) if len(dates) else None
    data_end = pd.Timestamp(dates[-1]) if len(dates) else None

    if crisis_windows and data_start is not None:
        # Keep only windows overlapping the plotted range (e.g. on the test
        # split only "2022 shock" and "Iran conflict" survive) and clip them,
        # so the historical spans don't blow up the x-axis.
        clipped = []
        for label, start, end in crisis_windows:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end) if end is not None else data_end
            if e < data_start or s > data_end:
                continue
            clipped.append((label, max(s, data_start), min(e, data_end)))
        if _shade is not None:
            # identical look to the section-3 plots; labels annotated once
            _shade(ax, clipped, annotate=True, data_end=data_end)
        else:  # fallback if section3_eda is unavailable
            for _, s, e in clipped:
                ax.axvspan(s, e, color="0.70", alpha=0.16, lw=0, zorder=0)

    ax.plot(dates, r, color="0.45", lw=0.7, alpha=0.9,
            label="Portfolio return", zorder=1)

    if var_parametric is not None:
        ax.plot(dates, np.asarray(var_parametric, dtype=float), lw=1.0,
                color="0.6", ls=":", label="Gaussian VaR", zorder=2)

    if es is not None:
        ax.plot(dates, np.asarray(es, dtype=float), lw=1.1, color="#7a0010",
                ls="--", label=f"FHS ES ({alpha:.0%})", zorder=3)

    ax.plot(dates, var, lw=1.3, color="#c62828",
            label=f"FHS VaR ({alpha:.0%})", zorder=4)

    ax.scatter(dates[hits], r[hits], s=14, color="#c62828",
               edgecolor="white", linewidth=0.4, zorder=5,
               label=f"Violations ({hits.sum()})")

    if data_start is not None:
        ax.set_xlim(data_start, data_end)

    hit_rate = hits.mean()
    ax.set_ylabel("Return / risk")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.text(0.99, 0.03,
            f"hit rate {hit_rate:.2%}  (target {alpha:.0%})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.3")
    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return hit_rate


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
