"""
Utility to calculate implied volatility using closed form Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm

N = norm.cdf


def bs_opt(F, K, T, vol, is_call):
    d1 = (np.log(F / K) + (0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if is_call:
        return F * N(d1) - K * N(d2)
    else:
        return K * N(-d2) - F * N(-d1)


def bs_vega(F, K, T, sigma):
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return F * N(d1) * np.sqrt(T)


def find_vol(target_value, F, K, T, is_call):
    if target_value < 1e-4 * F:
        return None
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_opt(F, K, T, sigma, is_call)
        vega = bs_vega(F, K, T, sigma)
        diff = target_value - price  # our root
        if abs(diff) < PRECISION:
            return sigma
        sigma = sigma + diff / vega  # f(x) / f'(x)
    return sigma
