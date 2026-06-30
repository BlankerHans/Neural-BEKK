import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist


def get_quantiles(alpha: float, dist: str = "gaussian", nu: float | None = None):
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

    if dist == "gaussian":
        return norm.ppf(alpha), norm.ppf(1.0 - alpha)

    if dist == "student_t":
        if nu is None:
            raise ValueError("nu is required for dist='student_t'")
        if nu <= 2:
            raise ValueError(f"nu must be > 2, got {nu}")
        return t_dist.ppf(alpha, df=nu), t_dist.ppf(1.0 - alpha, df=nu)

    raise ValueError(f"Unknown dist: {dist}")


def calc_var_bands(
    returns: np.ndarray,
    variance: np.ndarray,
    alpha: float,
    dist: str = "gaussian",
    nu: float | None = None,
    mu: np.ndarray | float | None = None,
    variance_kind: str = "scale",  # "scale" or "cov"
):
    """
    returns: (T,)
    variance: (T,) predicted variance path
    variance_kind:
      - "scale": variance already belongs to distribution scale (typical for student_nll parameterization)
      - "cov": variance is covariance variance; for Student-t it is converted to scale via (nu-2)/nu
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(variance, dtype=float)

    if r.shape[0] != v.shape[0]:
        raise ValueError(f"returns and variance length mismatch: {r.shape[0]} vs {v.shape[0]}")

    if mu is None:
        mu_t = np.zeros_like(r)
    else:
        mu_t = np.asarray(mu, dtype=float)
        if mu_t.shape == ():
            mu_t = np.full_like(r, float(mu_t))
        if mu_t.shape[0] != r.shape[0]:
            raise ValueError("mu length must match returns length")

    if dist == "student_t" and variance_kind == "cov":
        if nu is None or nu <= 2:
            raise ValueError("For Student-t with covariance input, nu > 2 is required")
        v = v * (nu - 2.0) / nu  # Cov -> scale variance

    v = np.clip(v, 1e-12, None)
    sigma_t = np.sqrt(v)

    q_lo, q_hi = get_quantiles(alpha=alpha, dist=dist, nu=nu)
    var_lo = mu_t + q_lo * sigma_t
    var_hi = mu_t + q_hi * sigma_t

    hits = r <= var_lo
    hit_rate = float(np.mean(hits))

    return {
        "returns": r,
        "var_lo": var_lo,
        "var_hi": var_hi,
        "hits": hits,
        "hit_rate": hit_rate,
    }


def fhs_var_es(z_hist_init, r_oos, vol_oos, alpha=0.01, window=500, min_vol=1e-12):
    """                                                                   
        Calculate Filtered Historical Simulation VaR and Expected Shortfall (Adcock et al., 2012).           
                                                                              
        Parameters:                                                           
        -----------                                                           
        z_hist_init : array-like                                              
            Initial historical residuals                                      
        r_oos : array-like                                                    
            Out-of-sample returns                                             
        vol_oos : array-like                                                  
            Out-of-sample volatilities                                        
        alpha : float, default=0.01                                           
            Confidence level (e.g., 0.01 for 99%)                             
        window : int, default=500                                             
            Rolling window size for historical data                           
        min_vol : float, default=1e-12                                        
            Minimum volatility to prevent division by zero                    
                                                                              
        Returns:                                                              
        --------                                                              
        tuple : (var, ex_sf, hits)                                            
            VaR values, Expected Shortfall values, and hit indicators         
        """ 
    
    hist = list(np.asarray(z_hist_init, dtype=float))
    if window is not None and len(hist) > window:
        hist = hist[-window:]

    r_oos = np.asarray(r_oos, dtype=float)
    vol_oos = np.asarray(vol_oos, dtype=float)

    if r_oos.ndim != 1 or vol_oos.ndim != 1:
        raise ValueError("r_oos and vol_oos must be 1D")

    if len(r_oos) != len(vol_oos):
        raise ValueError(
            f"r_oos and vol_oos must have same length, got {len(r_oos)} and {len(vol_oos)}"
        )

    var = np.empty_like(r_oos)
    ex_sf = np.empty_like(r_oos)
    hits = np.empty_like(r_oos, dtype=bool)

    for t, (r_t, s_t) in enumerate(zip(r_oos, vol_oos)):
        q_alpha = np.quantile(hist, alpha)      # historisches Residual-Quantil (F⁻¹(α) → negativ)
        var_t = s_t * q_alpha                   # mu_t=0; sonst + mu_t (= σ·F⁻¹(α) → negativ)
        es_t = s_t * np.mean(np.array(hist)[np.array(hist) <= q_alpha]) # Tail-Mean → negativ

        var[t] = var_t
        ex_sf[t] = es_t
        hits[t] = (r_t <= var_t)

        z_new = r_t / max(s_t, min_vol)         # neues gefiltertes Residuum
        hist.append(z_new)
        if window is not None and len(hist) > window:
            hist.pop(0)

    return var, ex_sf, hits
