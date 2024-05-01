"""
Constructs datasets for different models. (work in progress).
Intention is that these will be calibrated for a given date, from the same option prices
or implied volatility data. As of now the data below is not calibrated.

Details of model specific data api:
https://qablet-academy.github.io/intro/models/mc/
"""

from datetime import datetime

import numpy as np
from qablet_contracts.timetable import py_to_ts
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline

MC_PARAMS = {
    "PATHS": 100_000,
    "TIMESTEP": 1 / 250,
    "SEED": 1,
}


def assets_data():
    # data for 2013 August 14th

    ticker = "SPX"
    ccy = "USD"

    rate = 0.005
    spot = 1685.39
    div_rate = 0.02
    times = np.array([0.0, 5.0])
    rates = np.array([rate, rate])
    discount_data = ("ZERO_RATES", np.column_stack((times, rates)))

    fwds = spot * np.exp((rates - div_rate) * times)
    fwd_data = ("FORWARDS", np.column_stack((times, fwds)))

    return {ccy: discount_data, ticker: fwd_data}


def rbergomi_data():
    prc_dt = datetime(2013, 8, 14)

    # As of now the data below is not calibrated.
    H = 0.05
    x_vec = [0.025, 0.025]
    t_vec = [.0025, 2.0]
    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(prc_dt).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "rB": {
            "ASSET": "SPX",
            "ALPHA": H - 0.5,
            "RHO": -0.9,
            "XI": CubicSpline(t_vec, x_vec),
            "ETA": 2.3,
        },
    }


def heston_data():
    prc_dt = datetime(2013, 8, 14)

    # As of now the data below is not calibrated.
    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(prc_dt).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "HESTON": {
            "ASSET": "SPX",
            "INITIAL_VAR": 0.020,
            "LONG_VAR": 0.032,
            "VOL_OF_VAR": 0.88,
            "MEANREV": 1.5,
            "CORRELATION": -0.85,
        },
    }


def localvol_data():
    prc_dt = datetime(2013, 8, 14)

    # As of now the data below is not calibrated.
    times = [0.01, 0.2, 1.0]
    logstrikes = [-5.0, -0.5, -0.1, 0.0, 0.1, 0.5, 5.0]
    vols = np.array(
        [
            [2.713, 0.884, 0.442, 0.222, 0.032, 0.032, 0.032],
            [2.187, 0.719, 0.372, 0.209, 0.032, 0.032, 0.032],
            [1.237, 0.435, 0.264, 0.200, 0.101, 0.032, 0.032],
        ]
    )
    volinterp = RegularGridInterpolator(
        (times, logstrikes), vols, fill_value=None, bounds_error=False
    )

    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(prc_dt).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "LV": {"ASSET": "SPX", "VOL": volinterp},
    }
