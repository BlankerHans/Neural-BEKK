"""Return transformations used by the covariance-model data pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def approximate_treasury_return_from_yield(
    yield_percent: pd.Series,
    *,
    maturity_years: int = 10,
    coupon_frequency: int = 2,
    trading_days: int = 252,
) -> pd.Series:
    """Approximate daily total returns of a constant-maturity par bond.

    ``yield_percent`` is an annual yield quoted in percentage points, as in
    Yahoo Finance's ``^TNX`` series. The approximation combines daily carry
    with the standard modified-duration and convexity price effects:

        r_t = y_{t-1}/252 - D_mod,t-1 * delta_y_t
              + 0.5 * convexity_t-1 * delta_y_t**2.

    This produces an investable-return proxy, not an observed Treasury index
    return and not the excess holding-yield construction in Bollerslev, Engle,
    and Wooldridge (1988).
    """
    yields = yield_percent.astype(float) / 100.0
    previous_yield = yields.shift(1)
    delta_yield = yields.diff()

    periods = maturity_years * coupon_frequency
    period = np.arange(1, periods + 1, dtype=float)
    time_years = period / coupon_frequency

    result = pd.Series(np.nan, index=yields.index, dtype=float, name=yield_percent.name)
    valid = previous_yield.notna() & delta_yield.notna()

    for date in yields.index[valid]:
        y_prev = previous_yield.loc[date]
        periodic_yield = y_prev / coupon_frequency

        cash_flows = np.full(periods, periodic_yield, dtype=float)
        cash_flows[-1] += 1.0
        discount = (1.0 + periodic_yield) ** period
        present_values = cash_flows / discount
        price = present_values.sum()

        macaulay_duration = (time_years * present_values).sum() / price
        modified_duration = macaulay_duration / (1.0 + periodic_yield)
        convexity = (
            cash_flows
            * period
            * (period + 1.0)
            / (coupon_frequency**2 * (1.0 + periodic_yield) ** (period + 2.0))
        ).sum() / price

        dy = delta_yield.loc[date]
        result.loc[date] = (
            y_prev / trading_days
            - modified_duration * dy
            + 0.5 * convexity * dy**2
        )

    return result
