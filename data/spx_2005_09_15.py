"""
Model parameters in this file are as of 2005 September 15th.

See the README.md in this folder for details on the data.

The `_data` methods return a qablet dataset. See qablet dataset api in
https://qablet-academy.github.io/intro/models/mc/
"""

from datetime import datetime

import numpy as np
import pandas as pd
from qablet_contracts.timetable import py_to_ts
from scipy.interpolate import CubicSpline

MC_PARAMS = {
    "PATHS": 100_000,
    "TIMESTEP": 1 / 250,
    "SEED": 1,
}


def basic_info():
    """Common data we will need in all notebooks
    - pricing date, ticker, ccy and spot price"""

    return {
        "prc_dt": datetime(2005, 9, 14),
        "ticker": "SPX",
        "ccy": "USD",
        "spot": 1227.16,
    }


def assets_data():
    info = basic_info()

    rate = 0.045
    div_rate = 0.02
    times = np.array([0.0, 5.0])
    rates = np.array([rate, rate])
    discount_data = ("ZERO_RATES", np.column_stack((times, rates)))

    fwds = info["spot"] * np.exp((rates - div_rate) * times)
    fwd_data = ("FORWARDS", np.column_stack((times, fwds)))

    return {info["ccy"]: discount_data, info["ticker"]: fwd_data}


def rbergomi_data():
    """See script header for details."""
    info = basic_info()

    # time grid points for xi used in the NN
    tiny = 1e-6
    t_vec = np.concatenate(
        [
            np.arange(0.0025, 0.0175 + tiny, 0.0025),
            np.arange(0.02, 0.14 + tiny, 0.02),
            np.arange(0.16, 1 + tiny, 0.12),
            np.arange(1.25, 2 + tiny, 0.25),
            [3],
        ]
    )
    # parameters for xi
    xi_near, xi_far, xi_decay = 0.01313237, 0.02826082, 4.9101388
    xi_vec = xi_far + np.exp(-xi_decay * t_vec) * (xi_near - xi_far)

    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "rB": {
            "ASSET": "SPX",
            "ALPHA": 0.04130521 - 0.5,
            "RHO": -0.97397541,
            "XI": CubicSpline(t_vec, xi_vec),
            "ETA": 2.02721794,
        },
    }


def heston_data():
    info = basic_info()

    # See script header.
    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "HESTON": {
            "ASSET": "SPX",
            "INITIAL_VAR": 0.0174,
            "LONG_VAR": 0.0354,
            "VOL_OF_VOL": 0.3877,
            "MEANREV": 1.3253,
            "CORRELATION": -0.7165,
        },
    }


def localvol_data():
    info = basic_info()

    svidf = pd.read_csv("data/spx_svi_2005_09_15.csv")

    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "LV": {"ASSET": "SPX", "VOL": svidf},
    }
