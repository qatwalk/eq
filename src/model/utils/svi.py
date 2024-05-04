"""
Various SVI utilities - parameterizations and local volatility calculation.
"""

import numpy as np
import pandas as pd


from scipy.interpolate import RegularGridInterpolator


def svi(params, k):
    """Get total variance from SVI params and log strikes."""
    k_m = k - params["m"]
    discr = np.sqrt(k_m**2 + params["sig"] ** 2)
    w = params["a"] + params["b"] * (params["rho"] * k_m + discr)
    return w


def _svi_local_var_step(k, dk, t0, t1, w_vec_prev, w_vec):
    """Calculate local variance (vol**2) from t0 to t1, given a list of strikes k, uniformly spaced by dk,
    and the total variances (w = T * vol**2) at time t0 and t1.
    The result vector is two elements shorter than the input vectors k and w."""

    w = (w_vec + w_vec_prev) / 2
    # time derivative of w
    wt = (w_vec - w_vec_prev) / (t1 - t0)

    # strike derivatives of w
    wk = (w[2:] - w[:-2]) / (2 * dk)
    wkk = (w[2:] + w[:-2] - 2 * w[1:-1]) / (dk**2)

    # drop the extra points at top and bottom
    w_ = w[1:-1]
    k_ = k[1:-1]
    wt_ = wt[1:-1]

    # apply Dupire's formula
    return wt_ / (
        1
        - k_ / w_ * wk
        + 1 / 4 * (-1 / 4 - 1 / w_ + k_**2 / w_**2) * (wk) ** 2
        + 1 / 2 * wkk
    )


def get_w_interp(k_vec, svi_df):
    """Create an interpolator of total variances by t and k."""
    texp_vec = svi_df["texp"]

    ws = np.zeros((len(texp_vec) + 1, len(k_vec)))  # initialize

    # let ws[0] be zero, update ws[1:]
    for i, t in enumerate(texp_vec):
        ws[i + 1] = svi(svi_df.loc[i], k_vec)

    return RegularGridInterpolator(
        ([0.0] + texp_vec.to_list(), k_vec),
        ws,
        fill_value=None,
        bounds_error=False,
    )


def svi_local_vol(file_path, kmin, kmax, dk, t_vec):
    """Calculates local vol from an SVI parameterized implied vol surface."""

    # read csv file with svi params
    svi_df = pd.read_csv(file_path)

    # Create an interpolator of total variance by exp t_vec and k_vec.
    k_vec = np.arange(kmin - dk, kmax + dk + dk / 2, dk)
    w_interp = get_w_interp(k_vec, svi_df)

    # Create a mesh of local vol by t_vec and k_vec
    vols = np.zeros((len(t_vec) - 1, len(k_vec) - 2))
    w_vec_prev = np.zeros(len(k_vec))
    for i in range(len(t_vec) - 1):
        t0 = t_vec[i]
        t1 = t_vec[i + 1]
        w_vec = w_interp((t1, k_vec))
        lvar = _svi_local_var_step(k_vec, dk, t0, t1, w_vec_prev, w_vec)
        vols[i] = np.sqrt(lvar)
        w_vec_prev = w_vec

    return t_vec[1:], k_vec[1:-1], vols


if __name__ == "__main__":
    kmin, kmax, dk = -5.0, 5.0, 0.1
    t_vec = np.arange(0.0, 1.0, 0.1)
    times, strikes, vols = svi_local_vol(
        "data/spx_svi_2005_09_15.csv",
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        t_vec=t_vec,
    )
    mid = len(strikes) // 2
    print(vols[:, (0, mid, -1)])  # for each t: lowest K, atm, highest k
