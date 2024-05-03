"""
An example to try the rBergomi model with a Vanilla Option contract.
"""

from datetime import datetime

import numpy as np
from qablet_contracts.eq.vanilla import Option
from qablet_contracts.timetable import py_to_ts
from rbergomi import rBergomiMCModel
from scipy.interpolate import CubicSpline


def run_model():
    # Define the dataset
    times = np.array([0.0, 5.0])
    rates = np.array([0.04, 0.04])
    discount_data = ("ZERO_RATES", np.column_stack((times, rates)))
    spot = 4600
    div_rate = 0.01
    fwds = spot * np.exp((rates - div_rate) * times)
    fwd_data = ("FORWARDS", np.column_stack((times, fwds)))
    ticker = "SPX"

    dataset = {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(datetime(2023, 12, 31)).value,
        "ASSETS": {"USD": discount_data, ticker: fwd_data},
        "MC": {
            "PATHS": 100_000,
            "TIMESTEP": 1 / 250,
            "SEED": 1,
        },
        "rB": {
            "ASSET": ticker,
            "ALPHA": -0.45,
            "RHO": -0.8,
            "XI": CubicSpline([0.0025, 2.0], [0.025, 0.025]),
            "ETA": 2.5,
        },
    }

    # Create a Vanilla Option Contract
    opt_timetable = Option(
        "USD",
        ticker,
        strike=spot,
        maturity=datetime(2024, 12, 31),
        is_call=True,
    ).timetable()

    # Create model and price the option
    model = rBergomiMCModel()
    return model.price(opt_timetable, dataset)


if __name__ == "__main__":
    price, _ = run_model()
    print(f"price = {price}")
