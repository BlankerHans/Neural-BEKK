from statistics import NormalDist
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt

def calc_var(alpha, loss_fn, loss_kwargs=None):
    if loss_fn == gaussian_nll:
        z_lo = NormalDist().inv_cdf(alpha)        # z_{alpha}, z.B. -2.326
        z_hi = NormalDist().inv_cdf(1.0 - alpha)  # z_{1-alpha}, z.B. +2.326
    elif loss_fn == student_nll:
        nu = loss_kwargs.get("nu")
        z_lo = t.ppf(alpha, df=nu)
        z_hi = t.ppf(1-alpha, df=nu)

    return z_lo, z_hi