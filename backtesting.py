from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import chi2, t as student_t


def _as_1d_float_array(name: str, x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    return alpha


def _prepare_inputs(
    returns,
    var,
    es,
    volatility=None,
    dropna: bool = True,
):
    r = _as_1d_float_array("returns", returns)
    q = _as_1d_float_array("var", var)
    e = _as_1d_float_array("es", es)

    if not (len(r) == len(q) == len(e)):
        raise ValueError(
            f"returns, var and es must have same length, got {len(r)}, {len(q)}, {len(e)}"
        )

    s = None
    if volatility is not None:
        s = _as_1d_float_array("volatility", volatility)
        if len(s) != len(r):
            raise ValueError(
                f"volatility must have same length as returns, got {len(s)} vs {len(r)}"
            )

    if dropna:
        mask = np.isfinite(r) & np.isfinite(q) & np.isfinite(e)
        if s is not None:
            mask &= np.isfinite(s)
        r = r[mask]
        q = q[mask]
        e = e[mask]
        if s is not None:
            s = s[mask]

    if len(r) == 0:
        raise ValueError("No valid observations left after filtering non-finite values")

    return r, q, e, s


def _prepare_comparison_inputs(
    returns,
    var_1,
    es_1,
    var_2,
    es_2,
    dropna: bool = True,
):
    arrays = {
        "returns": _as_1d_float_array("returns", returns),
        "var_1": _as_1d_float_array("var_1", var_1),
        "es_1": _as_1d_float_array("es_1", es_1),
        "var_2": _as_1d_float_array("var_2", var_2),
        "es_2": _as_1d_float_array("es_2", es_2),
    }
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Comparison inputs must have same length, got {lengths}")

    if dropna:
        mask = np.ones(lengths["returns"], dtype=bool)
        for value in arrays.values():
            mask &= np.isfinite(value)
        arrays = {name: value[mask] for name, value in arrays.items()}

    if len(arrays["returns"]) == 0:
        raise ValueError("No valid observations left after filtering non-finite values")

    return (
        arrays["returns"],
        arrays["var_1"],
        arrays["es_1"],
        arrays["var_2"],
        arrays["es_2"],
    )


def _prepare_var_inputs(
    returns,
    var,
    dropna: bool = True,
):
    r = _as_1d_float_array("returns", returns)
    q = _as_1d_float_array("var", var)

    if len(r) != len(q):
        raise ValueError(f"returns and var must have same length, got {len(r)} and {len(q)}")

    if dropna:
        mask = np.isfinite(r) & np.isfinite(q)
        r = r[mask]
        q = q[mask]

    if len(r) == 0:
        raise ValueError("No valid observations left after filtering non-finite values")

    return r, q


def _var_hits(returns, var) -> np.ndarray:
    return np.asarray(returns, dtype=float) <= np.asarray(var, dtype=float)


def _as_lag_tuple(name: str, lags, allow_zero: bool = False) -> tuple[int, ...]:
    if lags is None:
        return ()
    if isinstance(lags, (bool, np.bool_)):
        raise ValueError(f"{name} must contain integer lags, got boolean")
    if isinstance(lags, (int, np.integer)):
        lags = (int(lags),)

    out = tuple(int(lag) for lag in lags)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicate lags: {out}")
    for lag in out:
        if lag < 0 or (lag == 0 and not allow_zero):
            lower = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must contain {lower} integer lags, got {out}")
    return out


def _safe_xlogy(count: int, prob: float) -> float:
    if count == 0:
        return 0.0
    if not (0.0 < prob <= 1.0):
        return -np.inf
    return float(count) * float(np.log(prob))


def _binomial_loglik(successes: int, total: int, prob: float) -> float:
    failures = int(total) - int(successes)
    return _safe_xlogy(failures, 1.0 - prob) + _safe_xlogy(successes, prob)


def _lr_pvalue(lr_stat: float, df: int) -> float:
    if not np.isfinite(lr_stat):
        return 0.0
    return float(chi2.sf(max(float(lr_stat), 0.0), df=df))


def _convert_r_value(x):
    try:
        n = len(x)
    except TypeError:
        return x

    if n == 1:
        value = x[0]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    out = []
    for value in x:
        if isinstance(value, (bool, np.bool_)):
            out.append(bool(value))
        elif isinstance(value, (int, np.integer)):
            out.append(int(value))
        else:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                out.append(str(value))
    return out


def _r_list_to_dict(x) -> dict[str, Any]:
    return {name: _convert_r_value(x.rx2(name)) for name in x.names}


@lru_cache(maxsize=1)
def _load_esback():
    """Python API for loading the esback R package.

    Raises:
        ImportError: _description_
        ImportError: _description_

    Returns:
        _type_: _description_
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import BoolVector, FloatVector, IntVector, ListVector, StrVector
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "Python package 'rpy2' is required for R backtests. "
            "Install it and make sure R itself is available."
        ) from exc

    try:
        esback = importr("esback")
    except Exception as exc:
        raise ImportError(
            "R package 'esback' is required. In R, run install.packages('esback')."
        ) from exc

    return ro, FloatVector, BoolVector, IntVector, ListVector, StrVector, esback


@lru_cache(maxsize=1)
def _load_segmgarch():
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import FloatVector, IntVector
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "Python package 'rpy2' is required for the segMGarch DQ backtest. "
            "Install it and make sure R itself is available."
        ) from exc

    try:
        segmgarch = importr("segMGarch")
    except Exception as exc:
        raise ImportError(
            "R package 'segMGarch' is required for the DQ backtest. "
            "In R, run install.packages('segMGarch')."
        ) from exc

    return ro, FloatVector, IntVector, segmgarch


def run_kupiec_test(
    returns,
    var,
    alpha: float,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Kupiec (1995) unconditional coverage test for VaR exceedances.
    """
    alpha = _validate_alpha(alpha)
    r, q = _prepare_var_inputs(returns, var, dropna=dropna)

    hits = _var_hits(r, q)
    n_obs = int(len(r))
    n_exceptions = int(np.sum(hits))
    hit_rate = float(n_exceptions / n_obs)

    ll_null = _binomial_loglik(successes=n_exceptions, total=n_obs, prob=alpha)
    ll_alt = _binomial_loglik(successes=n_exceptions, total=n_obs, prob=hit_rate)
    lr_uc = float(max(0.0, 2.0 * (ll_alt - ll_null)))
    p_value = _lr_pvalue(lr_uc, df=1)

    return {
        "test": "kupiec_uc",
        "alpha": alpha,
        "n_obs": n_obs,
        "n_exceptions": n_exceptions,
        "expected_exceptions": float(alpha * n_obs),
        "hit_rate": hit_rate,
        "lr_uc": lr_uc,
        "p_value": p_value,
    }


def run_christoffersen_test(
    returns,
    var,
    alpha: float,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Christoffersen (1998) independence and conditional coverage test for VaR exceedances.
    """
    alpha = _validate_alpha(alpha)
    r, q = _prepare_var_inputs(returns, var, dropna=dropna)

    hits = _var_hits(r, q).astype(int)
    n_obs = int(len(hits))
    if n_obs < 2:
        raise ValueError("Christoffersen test requires at least two observations")

    prev_hits = hits[:-1]
    next_hits = hits[1:]

    n00 = int(np.sum((prev_hits == 0) & (next_hits == 0)))
    n01 = int(np.sum((prev_hits == 0) & (next_hits == 1)))
    n10 = int(np.sum((prev_hits == 1) & (next_hits == 0)))
    n11 = int(np.sum((prev_hits == 1) & (next_hits == 1)))
    n_transitions = n00 + n01 + n10 + n11

    pi_hat = float(np.mean(hits))
    pi_01 = float(n01 / (n00 + n01)) if (n00 + n01) > 0 else np.nan
    pi_11 = float(n11 / (n10 + n11)) if (n10 + n11) > 0 else np.nan
    pi_pooled = float((n01 + n11) / n_transitions)

    ll_ind_null = _binomial_loglik(successes=n01 + n11, total=n_transitions, prob=pi_pooled)
    ll_ind_alt = _binomial_loglik(successes=n01, total=n00 + n01, prob=pi_01)
    ll_ind_alt += _binomial_loglik(successes=n11, total=n10 + n11, prob=pi_11)

    lr_ind = float(max(0.0, 2.0 * (ll_ind_alt - ll_ind_null)))
    p_value_ind = _lr_pvalue(lr_ind, df=1)

    ll_uc_null = _binomial_loglik(successes=int(np.sum(hits)), total=n_obs, prob=alpha)
    ll_uc_alt = _binomial_loglik(successes=int(np.sum(hits)), total=n_obs, prob=pi_hat)
    lr_uc = float(max(0.0, 2.0 * (ll_uc_alt - ll_uc_null)))
    lr_cc = float(max(0.0, lr_uc + lr_ind))
    p_value_cc = _lr_pvalue(lr_cc, df=2)

    return {
        "test": "christoffersen_cc",
        "alpha": alpha,
        "n_obs": n_obs,
        "n_exceptions": int(np.sum(hits)),
        "hit_rate": pi_hat,
        "transition_counts": {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
        },
        "transition_probs": {
            "pi_01": pi_01,
            "pi_11": pi_11,
            "pi_pooled": pi_pooled,
        },
        "lr_ind": lr_ind,
        "p_value_ind": p_value_ind,
        "lr_uc": lr_uc,
        "lr_cc": lr_cc,
        "p_value_cc": p_value_cc,
    }


def run_dq_test(
    returns,
    var,
    alpha: float,
    hit_lags=(1, 2, 3, 4),
    var_lags=(0,),
    return_lags=(),
    include_intercept: bool = True,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Engle-Manganelli dynamic quantile test for lower-tail VaR forecasts.

    Uses the R implementation ``segMGarch::DQtest``. That implementation always
    includes an intercept, the VaR forecast, lagged hits, and one lagged squared
    return term as regressors.
    """
    alpha = _validate_alpha(alpha)
    r, q = _prepare_var_inputs(returns, var, dropna=dropna)

    hit_lags = _as_lag_tuple("hit_lags", hit_lags, allow_zero=False)
    var_lags = _as_lag_tuple("var_lags", var_lags, allow_zero=True)
    return_lags = _as_lag_tuple("return_lags", return_lags, allow_zero=False)
    if not include_intercept:
        raise ValueError("segMGarch::DQtest always includes an intercept")
    if not hit_lags:
        raise ValueError("segMGarch::DQtest requires at least one lagged hit")
    expected_hit_lags = tuple(range(1, max(hit_lags) + 1))
    if hit_lags != expected_hit_lags:
        raise ValueError(
            "segMGarch::DQtest supports consecutive hit lags only, "
            f"expected {expected_hit_lags}, got {hit_lags}"
        )
    if var_lags != (0,):
        raise ValueError("segMGarch::DQtest always includes the current VaR forecast")
    if len(return_lags) > 1:
        raise ValueError("segMGarch::DQtest supports one squared-return lag only")

    seg_lag = int(return_lags[0]) if return_lags else 1
    seg_lag_hit = int(max(hit_lags))
    seg_lag_var = 1
    max_lag = max(seg_lag, seg_lag_hit, seg_lag_var)

    if len(r) <= max_lag:
        raise ValueError(
            f"DQ test requires more observations than max lag {max_lag}, got {len(r)}"
        )

    _, FloatVector, IntVector, segmgarch = _load_segmgarch()
    result = segmgarch.DQtest(
        y=FloatVector(r),
        VaR=FloatVector(q),
        VaR_level=FloatVector([1.0 - alpha]),
        lag=IntVector([seg_lag]),
        lag_hit=IntVector([seg_lag_hit]),
        lag_var=IntVector([seg_lag_var]),
    )

    dq_stat = float(np.asarray(result[0], dtype=float).reshape(-1)[0])
    p_value = float(np.asarray(result[1], dtype=float).reshape(-1)[0])
    df = int(seg_lag_hit + 3)

    aligned_hits = _var_hits(r, q).astype(float)[-int(len(r) - max_lag) :]
    return {
        "test": "dynamic_quantile",
        "implementation": "segMGarch::DQtest",
        "alpha": alpha,
        "var_level": float(1.0 - alpha),
        "n_obs": int(len(aligned_hits)),
        "n_exceptions": int(np.sum(aligned_hits)),
        "expected_exceptions": float(alpha * len(aligned_hits)),
        "hit_rate": float(np.mean(aligned_hits)),
        "max_lag": int(max_lag),
        "df": df,
        "dq_stat": dq_stat,
        "p_value": p_value,
        "columns": [
            "intercept",
            "var_forecast",
            *[f"hit_lag{i}" for i in range(1, seg_lag_hit + 1)],
            f"return_sq_lag{seg_lag}",
        ],
        "segmgarch_lags": {
            "lag": seg_lag,
            "lag_hit": seg_lag_hit,
            "lag_var": seg_lag_var,
        },
    }


def calculate_fz_loss(
    returns,
    var,
    es,
    alpha: float,
    dropna: bool = True,
) -> np.ndarray:
    """
    Fissler-Ziegel joint VaR-ES loss in the Patton-Ziegel-Chen parameterization.

    This matches ``GAS::FZLoss`` for lower-tail return forecasts:
    ``I(r <= VaR) * (r - VaR) / (alpha * ES) + VaR / ES + log(-ES) - 1``.
    """
    alpha = _validate_alpha(alpha)
    r, q, e, _ = _prepare_inputs(returns, var, es, volatility=None, dropna=dropna)

    if np.any(e >= 0.0):
        raise ValueError("FZ loss requires negative ES forecasts because it uses log(-ES)")

    hits = _var_hits(r, q).astype(float)
    loss = hits * (r - q) / (alpha * e) + q / e + np.log(-e) - 1.0
    return loss


def build_fz_loss_matrix(
    model_forecasts: dict[str, dict[str, Any]],
    alpha: float,
    dropna: bool = True,
) -> tuple[tuple[str, ...], np.ndarray]:
    """
    Build an aligned T x M FZ-loss matrix for MCS input.

    Each forecast entry must contain ``returns``, ``var`` and ``es``. All
    models must be evaluated on the same realized return path.
    """
    alpha = _validate_alpha(alpha)
    model_names = tuple(model_forecasts.keys())
    if not model_names:
        raise ValueError("At least one model forecast is required")

    base_returns = None
    valid_mask = None
    losses = []
    for model_name in model_names:
        forecast = model_forecasts[model_name]
        missing = {"returns", "var", "es"} - set(forecast)
        if missing:
            raise ValueError(f"Model {model_name!r} is missing fields: {sorted(missing)}")

        returns = _as_1d_float_array(f"{model_name}.returns", forecast["returns"])
        var = _as_1d_float_array(f"{model_name}.var", forecast["var"])
        es = _as_1d_float_array(f"{model_name}.es", forecast["es"])
        if not (len(returns) == len(var) == len(es)):
            raise ValueError(
                f"Model {model_name!r} inputs must have same length, "
                f"got returns={len(returns)}, var={len(var)}, es={len(es)}"
            )

        if base_returns is None:
            base_returns = returns
        elif len(returns) != len(base_returns) or not np.allclose(
            returns,
            base_returns,
            equal_nan=True,
        ):
            raise ValueError(
                f"Returns for model {model_name!r} are not aligned; "
                "FZ-loss MCS requires the same realized returns."
            )

        model_valid = np.isfinite(returns) & np.isfinite(var) & np.isfinite(es)
        valid_mask = model_valid if valid_mask is None else valid_mask & model_valid
        losses.append(
            calculate_fz_loss(
                returns=returns,
                var=var,
                es=es,
                alpha=alpha,
                dropna=False,
            )
        )

    loss_matrix = np.column_stack(losses)
    if dropna:
        valid_mask = valid_mask & np.isfinite(loss_matrix).all(axis=1)
        loss_matrix = loss_matrix[valid_mask]

    if len(loss_matrix) == 0:
        raise ValueError("No valid observations left after filtering non-finite FZ losses")

    return model_names, loss_matrix


def _long_run_variance_acf(x: np.ndarray, max_lag: int) -> float:
    centered = x - np.mean(x)
    n_obs = len(centered)
    gamma_0 = float(np.dot(centered, centered) / n_obs)
    autocovariances = [
        float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        for lag in range(1, max_lag + 1)
    ]
    return float(gamma_0 + 2.0 * sum(autocovariances))


def run_modified_dm_test(
    loss_differential,
    h: int = 1,
    cl: float = 0.05,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Harvey-Leybourne-Newbold modified Diebold-Mariano test.

    ``loss_differential`` must be ``loss_model_1 - loss_model_2``.
    Negative means therefore favor model 1.
    """
    if isinstance(h, (bool, np.bool_)):
        raise ValueError("h must be a positive integer, got boolean")
    h = int(h)
    if h < 1:
        raise ValueError(f"h must be a positive integer, got {h}")

    cl = float(cl)
    if not (0.0 < cl < 1.0):
        raise ValueError(f"cl must be in (0, 1), got {cl}")

    d = _as_1d_float_array("loss_differential", loss_differential)
    if dropna:
        d = d[np.isfinite(d)]
    if len(d) <= h:
        raise ValueError(f"DM test requires more observations than h={h}, got {len(d)}")

    n_obs = int(len(d))
    mean_d = float(np.mean(d))
    long_run_variance = _long_run_variance_acf(d, max_lag=h - 1)
    if not np.isfinite(long_run_variance) or long_run_variance <= 0.0:
        raise ValueError(
            "DM test requires a positive long-run variance of the loss differential"
        )

    dm_stat = float(mean_d / np.sqrt(long_run_variance / n_obs))
    hln_term = (n_obs + 1.0 - 2.0 * h + h * (h - 1.0) / n_obs) / n_obs
    if hln_term <= 0.0:
        raise ValueError(
            f"HLN correction is not defined for n_obs={n_obs} and h={h}"
        )

    hln_correction = float(np.sqrt(hln_term))
    modified_dm_stat = float(dm_stat * hln_correction)
    df = int(n_obs - 1)
    p_value = float(2.0 * student_t.sf(abs(modified_dm_stat), df=df))
    reject = bool(p_value < cl)

    return {
        "test": "modified_diebold_mariano",
        "implementation": "python_harvey_leybourne_newbold_1997",
        "alternative": "two-sided",
        "h": h,
        "cl": cl,
        "n_obs": n_obs,
        "df": df,
        "mean_loss_differential": mean_d,
        "long_run_variance": long_run_variance,
        "dm_stat": dm_stat,
        "hln_correction": hln_correction,
        "modified_dm_stat": modified_dm_stat,
        "stat": modified_dm_stat,
        "p_value": p_value,
        "pval": p_value,
        "reject": reject,
        "rej": reject,
        "loss_differential": "loss_model_1_minus_loss_model_2",
    }


def run_fz_dm_test(
    returns,
    var_1,
    es_1,
    var_2,
    es_2,
    alpha: float,
    h: int = 1,
    cl: float = 0.05,
    model_1: str = "model_1",
    model_2: str = "model_2",
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Compare two VaR-ES forecast paths with FZ loss and the HLN modified DM test.
    """
    alpha = _validate_alpha(alpha)
    r, q1, e1, q2, e2 = _prepare_comparison_inputs(
        returns=returns,
        var_1=var_1,
        es_1=es_1,
        var_2=var_2,
        es_2=es_2,
        dropna=dropna,
    )

    loss_1 = calculate_fz_loss(r, q1, e1, alpha=alpha, dropna=False)
    loss_2 = calculate_fz_loss(r, q2, e2, alpha=alpha, dropna=False)
    loss_differential = loss_1 - loss_2
    result = run_modified_dm_test(
        loss_differential=loss_differential,
        h=h,
        cl=cl,
        dropna=False,
    )

    mean_loss_1 = float(np.mean(loss_1))
    mean_loss_2 = float(np.mean(loss_2))
    if mean_loss_1 < mean_loss_2:
        preferred = model_1
    elif mean_loss_2 < mean_loss_1:
        preferred = model_2
    else:
        preferred = "tie"

    result.update(
        {
            "test": "fz_modified_diebold_mariano",
            "implementation": "python_fz_loss + python_hln_modified_dm",
            "loss_function": "fissler_ziegel_patton_ziegel_chen",
            "dm_implementation": "python_harvey_leybourne_newbold_1997",
            "alpha": alpha,
            "model_1": str(model_1),
            "model_2": str(model_2),
            "mean_loss_model_1": mean_loss_1,
            "mean_loss_model_2": mean_loss_2,
            "preferred_by_mean_loss": preferred,
        }
    )
    return result


def run_fz_dm_comparison_plan(
    model_forecasts: dict[str, dict[str, Any]],
    alpha: float,
    benchmark: str = "bekk_symmetric",
    structured_pairs: tuple[tuple[str, str], ...] = (),
    include_pairwise: bool = True,
    h: int = 1,
    cl: float = 0.05,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Run the paper's pre-specified FZ-loss DM comparison plan.

    Each ``model_forecasts`` value must contain ``returns``, ``var`` and ``es``.
    Pair direction is always ``model_1 - model_2``. Negative mean loss
    differentials therefore favor the first model in the pair.
    """
    alpha = _validate_alpha(alpha)
    model_names = tuple(model_forecasts.keys())
    if benchmark not in model_forecasts:
        raise ValueError(f"Benchmark {benchmark!r} is not in model_forecasts")

    def run_pair(model_1: str, model_2: str) -> dict[str, Any]:
        if model_1 not in model_forecasts:
            raise ValueError(f"Unknown model in comparison plan: {model_1!r}")
        if model_2 not in model_forecasts:
            raise ValueError(f"Unknown model in comparison plan: {model_2!r}")

        forecast_1 = model_forecasts[model_1]
        forecast_2 = model_forecasts[model_2]
        missing_1 = {"returns", "var", "es"} - set(forecast_1)
        missing_2 = {"returns", "var", "es"} - set(forecast_2)
        if missing_1:
            raise ValueError(f"Model {model_1!r} is missing fields: {sorted(missing_1)}")
        if missing_2:
            raise ValueError(f"Model {model_2!r} is missing fields: {sorted(missing_2)}")

        returns_1 = _as_1d_float_array(f"{model_1}.returns", forecast_1["returns"])
        returns_2 = _as_1d_float_array(f"{model_2}.returns", forecast_2["returns"])
        if len(returns_1) != len(returns_2) or not np.allclose(
            returns_1,
            returns_2,
            equal_nan=True,
        ):
            raise ValueError(
                f"Returns for {model_1!r} and {model_2!r} are not aligned; "
                "DM comparisons require the same realized returns."
            )

        return run_fz_dm_test(
            returns=returns_1,
            var_1=forecast_1["var"],
            es_1=forecast_1["es"],
            var_2=forecast_2["var"],
            es_2=forecast_2["es"],
            alpha=alpha,
            h=h,
            cl=cl,
            model_1=model_1,
            model_2=model_2,
            dropna=dropna,
        )

    benchmark_results = {
        f"{model}_vs_{benchmark}": run_pair(model, benchmark)
        for model in model_names
        if model != benchmark
    }
    structured_results = {
        f"{model_1}_vs_{model_2}": run_pair(model_1, model_2)
        for model_1, model_2 in structured_pairs
    }

    pairwise_results = {}
    if include_pairwise:
        for i, model_1 in enumerate(model_names):
            for model_2 in model_names[i + 1 :]:
                pairwise_results[f"{model_1}_vs_{model_2}"] = run_pair(model_1, model_2)

    return {
        "meta": {
            "alpha": alpha,
            "h": int(h),
            "cl": float(cl),
            "benchmark": benchmark,
            "models": list(model_names),
            "structured_pairs": [list(pair) for pair in structured_pairs],
            "include_pairwise": bool(include_pairwise),
            "pair_direction": "model_1_minus_model_2",
            "negative_mean_loss_differential_favors": "model_1",
            "interpretation": "benchmark and structured comparisons are primary; pairwise comparisons are for appendix/robustness",
        },
        "benchmark": benchmark_results,
        "structured": structured_results,
        "pairwise": pairwise_results,
    }


def run_cc_backtest(
    returns,
    var,
    es,
    alpha: float,
    volatility=None,
    hommel: bool = True,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Conditional calibration backtest from the R package esback.

    This is an ES backtest and not the Christoffersen VaR conditional
    coverage test.

    Assumes returns, VaR and ES are aligned and expressed on the same return scale.
    """
    alpha = _validate_alpha(alpha)
    r, q, e, s = _prepare_inputs(returns, var, es, volatility=volatility, dropna=dropna)
    _, FloatVector, BoolVector, _, _, _, esback = _load_esback()

    kwargs = {
        "r": FloatVector(r),
        "q": FloatVector(q),
        "e": FloatVector(e),
        "alpha": FloatVector([alpha]),
        "hommel": BoolVector([hommel]),
    }
    if s is not None:
        kwargs["s"] = FloatVector(s)

    result = esback.cc_backtest(**kwargs)
    out = _r_list_to_dict(result)
    out["n_obs"] = int(len(r))
    return out


def run_er_backtest(
    returns,
    var,
    es,
    alpha: float,
    volatility=None,
    B: int = 1000,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Exceedance residual backtest from the R package esback.
    """
    alpha = _validate_alpha(alpha)
    r, q, e, s = _prepare_inputs(returns, var, es, volatility=volatility, dropna=dropna)
    _, FloatVector, _, IntVector, _, _, esback = _load_esback()

    kwargs = {
        "r": FloatVector(r),
        "q": FloatVector(q),
        "e": FloatVector(e),
        "B": IntVector([int(B)]),
    }
    if s is not None:
        kwargs["s"] = FloatVector(s)

    result = esback.er_backtest(**kwargs)
    out = _r_list_to_dict(result)
    out["n_obs"] = int(len(r))
    return out


def run_esr_backtest(
    returns,
    var,
    es,
    alpha: float,
    version: int = 1,
    B: int = 0,
    cov_config: dict[str, Any] | None = None,
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Expected shortfall regression backtest from the R package esback.

    version:
      1 = strict ESR
      2 = auxiliary ESR
      3 = strict intercept
    """
    alpha = _validate_alpha(alpha)
    r, q, e, _ = _prepare_inputs(returns, var, es, volatility=None, dropna=dropna)
    _, FloatVector, BoolVector, IntVector, ListVector, StrVector, esback = _load_esback()

    if version not in (1, 2, 3):
        raise ValueError(f"version must be one of (1, 2, 3), got {version}")

    if cov_config is None:
        cov_config = {
            "sparsity": "iid",
            "sigma_est": "scl_sp",
            "misspec": True,
        }

    r_cov_config = ListVector(
        {
            "sparsity": StrVector([str(cov_config.get("sparsity", "nid"))]),
            "sigma_est": StrVector([str(cov_config.get("sigma_est", "scl_sp"))]),
            "misspec": BoolVector([bool(cov_config.get("misspec", False))]),
        }
    )

    result = esback.esr_backtest(
        r=FloatVector(r),
        q=FloatVector(q),
        e=FloatVector(e),
        alpha=FloatVector([alpha]),
        version=IntVector([int(version)]),
        B=IntVector([int(B)]),
        cov_config=r_cov_config,
    )
    out = _r_list_to_dict(result)
    out["version"] = int(version)
    out["n_obs"] = int(len(r))
    out["cov_config"] = {
        "sparsity": str(cov_config.get("sparsity", "iid")),
        "sigma_est": str(cov_config.get("sigma_est", "scl_sp")),
        "misspec": bool(cov_config.get("misspec", False)),
    }
    return out


def run_backtest_suite(
    returns,
    var,
    es,
    alpha: float,
    volatility=None,
    hommel: bool = True,
    er_B: int = 1000,
    esr_B: int = 0,
    esr_versions: tuple[int, ...] = (1, 2, 3),
    cov_config: dict[str, Any] | None = None,
    dq_hit_lags=(1, 2, 3, 4),
    dq_var_lags=(0,),
    dq_return_lags=(),
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Run a compact suite of VaR and ES backtests and return Python-native results.

    Example:
        results = run_backtest_suite(
            returns=portfolio_test["r_p"].values,
            var=var_fhs_test,
            es=es_fhs_test,
            alpha=0.01,
            volatility=portfolio_test["vol_p"].values,
        )
    """
    alpha = _validate_alpha(alpha)
    r, q, e, s = _prepare_inputs(returns, var, es, volatility=volatility, dropna=dropna)

    results = {
        "meta": {
            "alpha": alpha,
            "n_obs": int(len(r)),
        },
        "var": {
            "kupiec": run_kupiec_test(
                returns=r,
                var=q,
                alpha=alpha,
                dropna=False,
            ),
            "christoffersen": run_christoffersen_test(
                returns=r,
                var=q,
                alpha=alpha,
                dropna=False,
            ),
            "dq": run_dq_test(
                returns=r,
                var=q,
                alpha=alpha,
                hit_lags=dq_hit_lags,
                var_lags=dq_var_lags,
                return_lags=dq_return_lags,
                dropna=False,
            ),
        },
        "cc": run_cc_backtest(
            returns=r,
            var=q,
            es=e,
            alpha=alpha,
            volatility=s,
            hommel=hommel,
            dropna=False,
        ),
        "er": run_er_backtest(
            returns=r,
            var=q,
            es=e,
            alpha=alpha,
            volatility=s,
            B=er_B,
            dropna=False,
        ),
        "esr": {},
    }

    for version in esr_versions:
        try:
            results["esr"][f"v{version}"] = run_esr_backtest(
                returns=r,
                var=q,
                es=e,
                alpha=alpha,
                version=version,
                B=esr_B,
                cov_config=cov_config,
                dropna=False,
            )
        except Exception as exc:
            results["esr"][f"v{version}"] = {
                "version": int(version),
                "n_obs": int(len(r)),
                "error": str(exc),
                "cov_config": cov_config
                if cov_config is not None
                else {"sparsity": "iid", "sigma_est": "scl_sp", "misspec": True},
            }

    return results
