"""
Model parameters in this file are as of 2005 September 15th.
The SVI parameters, and heston parameters are from Gatheral.
The local vol parameters are derived from the SVI parameters.
The Bergomi parameters are not calibrated yet.

Details of model specific data api:
https://qablet-academy.github.io/intro/models/mc/
"""

from datetime import datetime

import numpy as np
from qablet_contracts.timetable import py_to_ts
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from src.model.utils.svi import svi_local_vol

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
    info = basic_info()

    # As of now the data below is not calibrated.
    H = 0.05
    xi_vec = [0.0175, 0.021]
    t_vec = [0.0025, 0.10]
    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "rB": {
            "ASSET": "SPX",
            "ALPHA": H - 0.5,
            "RHO": -0.9,
            "XI": CubicSpline(t_vec, xi_vec),
            "ETA": 2.3,
        },
    }


def heston_data():
    info = basic_info()

    # Heston Parameters from Gatheral.
    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "HESTON": {
            "ASSET": "SPX",
            "INITIAL_VAR": 0.0174,
            "LONG_VAR": 0.0354,
            "VOL_OF_VAR": 0.3877,
            "MEANREV": 1.3253,
            "CORRELATION": -0.7165,
        },
    }


def localvol_data():
    info = basic_info()

    # SVI Parameters are from Gatheral.
    kmin, kmax, dk = -5.0, 5.0, 0.05
    t_vec = np.concatenate(
        (
            np.arange(0.0, 0.05 - 1e-6, 0.01),
            np.arange(0.05, 0.2 - 1e-6, 0.025),
            np.arange(0.2, 1.5 - 1e-6, 0.1),
        )
    )
    times, strikes, vols = svi_local_vol(
        "data/spx_svi_2005_09_15.csv",
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        t_vec=t_vec,
    )

    volinterp = RegularGridInterpolator(
        (times, strikes), vols, fill_value=None, bounds_error=False
    )

    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(),
        "MC": MC_PARAMS,
        "LV": {"ASSET": "SPX", "VOL": volinterp},
    }
