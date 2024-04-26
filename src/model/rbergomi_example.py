import pyarrow as pa
import numpy as np
from qablet_contracts.timetable import py_to_ts, TS_EVENT_SCHEMA
from datetime import datetime
from rbergomi import rBergomiMCModel


def run_model():
    model = rBergomiMCModel()

    times = np.array([0.0, 5.0])
    rates = np.array([0.04, 0.04])
    discount_data = ("ZERO_RATES", np.column_stack((times, rates)))
    div_rate = 0.01
    fwds = 100 * np.exp((rates - div_rate) * times)
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
            "ASSET": "SPX",
            "ALPHA": -0.45,
            "RHO": -0.8,
            "XI": 0.11,
            "ETA": 2.5,
        },
    }

    # We will define a forward timetable, instead of using contract classes from qablet_contracts
    events = [
        {
            "track": "",
            "time": datetime(2024, 12, 31),
            "op": "+",
            "quantity": 1,
            "unit": ticker,
        }
    ]

    events_table = pa.RecordBatch.from_pylist(events, schema=TS_EVENT_SCHEMA)
    fwd_timetable = {"events": events_table, "expressions": {}}

    return model.price(fwd_timetable, dataset)


if __name__ == "__main__":
    price, _ = run_model()
    print(f"price = {price}")
