"""
Utility to calculate implied volatility using Black-Scholes model.
"""

import numpy as np
import pyarrow as pa
from qablet.base.utils import Forwards, discounter_from_dataset
from qablet_contracts.timetable import TS_EVENT_SCHEMA, py_to_ts

from src.model.utils.bs import find_vol


def iv_surface(ticker, model, dataset, logstrikes, expirations):
    """
    Construct a volatility surface using provided qablet model and dataset as follows:

    - define a contract which is a series of forwards paying at given expiration dates.
    - use the cashflow stats to get the cashflow values at all paths
    - reconstrcut option prices for different strikes and maturities.
    - get implied vols from the option prices.
    """
    # Create a timetable that pays forwards at given expirations
    events = [
        {
            "track": "",
            "time": dt,
            "op": "+",
            "quantity": 1,
            "unit": ticker,
        }
        for dt in expirations
    ]

    events_table = pa.RecordBatch.from_pylist(events, schema=TS_EVENT_SCHEMA)
    fwd_timetable = {"events": events_table, "expressions": {}}

    discounter = discounter_from_dataset(dataset)

    _, stats = model.price(fwd_timetable, dataset)
    # cashflows for track 0, all events
    cf = stats["CASHFLOW"][0]

    asset_fwds = Forwards(dataset["ASSETS"][ticker])

    iv_mat = np.zeros((len(expirations), len(logstrikes)))
    for i, exp in enumerate(expirations):
        prc_ts = dataset["PRICING_TS"]
        # Get Time in years from the millisecond timestamps
        T = (py_to_ts(exp).value - prc_ts) / (365.25 * 24 * 3600 * 1e3)
        df = discounter.discount(T)
        fwd = asset_fwds.forward(T)
        strikes = fwd * np.exp(logstrikes)

        # Use a call option for strikes above forward, a put option otherwise
        is_call = strikes > fwd
        is_call_c = is_call[..., None]  # Turn into a column vector

        # calculate prices (value as of expiration date)
        event_cf = cf[i] / df
        strikes_c = strikes[..., None]  # Turn into a column vector
        pay = np.where(is_call_c, event_cf - strikes_c, strikes_c - event_cf)
        prices = np.maximum(pay, 0).mean(axis=1)

        # calculate implied vols
        iv_mat[i, :] = [
            find_vol(p, fwd, k, T, ic)
            for p, k, ic in zip(prices, strikes, is_call)
        ]

    return iv_mat
