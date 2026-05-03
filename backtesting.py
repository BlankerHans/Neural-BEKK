from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import chi2


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


def _make_lagged_design(
    series_by_name: dict[str, np.ndarray],
    lag_specs: dict[str, tuple[int, ...]],
    include_intercept: bool,
    start: int,
) -> tuple[np.ndarray, list[str]]:
    n_obs = len(next(iter(series_by_name.values())))
    cols = []
    names = []

    if include_intercept:
        cols.append(np.ones(n_obs - start, dtype=float))
        names.append("intercept")

    for series_name, values in series_by_name.items():
        for lag in lag_specs.get(series_name, ()):
            if lag == 0:
                cols.append(values[start:n_obs])
            else:
                cols.append(values[start - lag : n_obs - lag])
            names.append(f"{series_name}_lag{lag}")

    if not cols:
        raise ValueError("At least one instrument is required")

    return np.column_stack(cols), names


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

    hits = r < q
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

    hits = (r < q).astype(int)
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

    The test regresses centered exceedance hits on instruments known when the
    VaR forecast is made. The default specification matches the common CAViaR
    setup: intercept, current VaR forecast and four lagged hits.
    """
    alpha = _validate_alpha(alpha)
    r, q = _prepare_var_inputs(returns, var, dropna=dropna)

    hit_lags = _as_lag_tuple("hit_lags", hit_lags, allow_zero=False)
    var_lags = _as_lag_tuple("var_lags", var_lags, allow_zero=True)
    return_lags = _as_lag_tuple("return_lags", return_lags, allow_zero=False)
    max_lag = max(hit_lags + var_lags + return_lags + (0,))

    if len(r) <= max_lag:
        raise ValueError(
            f"DQ test requires more observations than max lag {max_lag}, got {len(r)}"
        )

    hits = (r < q).astype(float)
    centered_hits = hits - alpha
    y = centered_hits[max_lag:]
    x, column_names = _make_lagged_design(
        series_by_name={
            "var": q,
            "hit": centered_hits,
            "return": r,
        },
        lag_specs={
            "var": var_lags,
            "hit": hit_lags,
            "return": return_lags,
        },
        include_intercept=include_intercept,
        start=max_lag,
    )

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = np.linalg.pinv(x) @ y
    dq_stat = float(y @ x @ xtx_inv @ x.T @ y / (alpha * (1.0 - alpha)))
    df = int(np.linalg.matrix_rank(x))
    p_value = _lr_pvalue(dq_stat, df=df)

    aligned_hits = hits[max_lag:]
    return {
        "test": "dynamic_quantile",
        "alpha": alpha,
        "n_obs": int(len(y)),
        "n_exceptions": int(np.sum(aligned_hits)),
        "expected_exceptions": float(alpha * len(y)),
        "hit_rate": float(np.mean(aligned_hits)),
        "max_lag": int(max_lag),
        "df": df,
        "dq_stat": dq_stat,
        "p_value": p_value,
        "columns": column_names,
        "coefficients": {
            name: float(value) for name, value in zip(column_names, beta, strict=True)
        },
    }


def run_dynamic_calibration_test(
    returns,
    var,
    es,
    alpha: float,
    hit_lags=(1, 2, 3, 4),
    var_lags=(0,),
    es_lags=(0,),
    return_lags=(),
    include_intercept: bool = True,
    components=("var", "es"),
    dropna: bool = True,
) -> dict[str, Any]:
    """
    DQ-style dynamic calibration test for joint lower-tail VaR/ES forecasts.

    This is not the Engle-Manganelli VaR-only DQ test. It tests whether the
    VaR/ES identification functions are orthogonal to the chosen instruments.
    """
    alpha = _validate_alpha(alpha)
    r, q, e, _ = _prepare_inputs(returns, var, es, volatility=None, dropna=dropna)

    hit_lags = _as_lag_tuple("hit_lags", hit_lags, allow_zero=False)
    var_lags = _as_lag_tuple("var_lags", var_lags, allow_zero=True)
    es_lags = _as_lag_tuple("es_lags", es_lags, allow_zero=True)
    return_lags = _as_lag_tuple("return_lags", return_lags, allow_zero=False)
    max_lag = max(hit_lags + var_lags + es_lags + return_lags + (0,))

    if len(r) <= max_lag:
        raise ValueError(
            "Dynamic calibration test requires more observations than "
            f"max lag {max_lag}, got {len(r)}"
        )

    hits = (r <= q).astype(float)
    identification = {
        "var": alpha - hits,
        "es": e - q + hits * (q - r) / alpha,
    }
    components = tuple(str(component) for component in components)
    unknown_components = set(components) - set(identification)
    if unknown_components:
        raise ValueError(f"Unknown components: {sorted(unknown_components)}")
    if not components:
        raise ValueError("At least one component is required")

    v = np.column_stack([identification[component][max_lag:] for component in components])
    x, column_names = _make_lagged_design(
        series_by_name={
            "var": q,
            "es": e,
            "hit": hits - alpha,
            "return": r,
        },
        lag_specs={
            "var": var_lags,
            "es": es_lags,
            "hit": hit_lags,
            "return": return_lags,
        },
        include_intercept=include_intercept,
        start=max_lag,
    )

    moments = np.einsum("ti,tj->tij", x, v).reshape(len(v), -1)
    moment_mean = np.mean(moments, axis=0)
    omega = moments.T @ moments / len(v)
    stat = float(len(v) * moment_mean @ np.linalg.pinv(omega) @ moment_mean)
    df = int(np.linalg.matrix_rank(omega))
    p_value = _lr_pvalue(stat, df=df)

    moment_names = [
        f"{column}:{component}" for column in column_names for component in components
    ]
    return {
        "test": "dynamic_joint_calibration",
        "alpha": alpha,
        "n_obs": int(len(v)),
        "max_lag": int(max_lag),
        "df": df,
        "stat": stat,
        "p_value": p_value,
        "columns": column_names,
        "components": list(components),
        "moment_means": {
            name: float(value)
            for name, value in zip(moment_names, moment_mean, strict=True)
        },
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
    dq_es_lags=(0,),
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
        "dynamic_calibration": run_dynamic_calibration_test(
            returns=r,
            var=q,
            es=e,
            alpha=alpha,
            hit_lags=dq_hit_lags,
            var_lags=dq_var_lags,
            es_lags=dq_es_lags,
            return_lags=dq_return_lags,
            dropna=False,
        ),
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
