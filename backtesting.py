from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


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
    dropna: bool = True,
) -> dict[str, Any]:
    """
    Run a compact suite of esback tests and return Python-native results.

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
