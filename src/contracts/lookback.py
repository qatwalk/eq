"""
Create a timetable for a look-back put option contract.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
from qablet_contracts.timetable import TS_EVENT_SCHEMA


# Creating a look-back put option timetable with custom lookbacks
def lookback_put_timetable(ticker, start_date, T, num_points):
    start_date = pd.to_datetime(start_date)
    days_to_maturity = T * 365.25
    maturity = start_date + timedelta(days=days_to_maturity)

    # find fixing dates, including the start date, but not the maturity date.
    fix_dates = pd.date_range(
        start=start_date,
        end=maturity,
        periods=num_points + 1,
        inclusive="left",
    )
    events = [
        {
            "track": "",
            "time": fix_dates[0],
            "op": None,
            "quantity": 0,
            "unit": "INIT",
        }
    ]

    for fixing_time in fix_dates[1:]:
        events.append(
            {
                "track": "",
                "time": fixing_time,
                "op": None,
                "quantity": 0,
                "unit": "UPDATE",
            }
        )

    events.append(
        {
            "track": "",
            "time": maturity,
            "op": "+",
            "quantity": 1,
            "unit": "LOOKBACK",
        }
    )

    # Defining fixed strike look-back put payoff function
    def lookback_put_pay_fn(inputs):
        [ticker, s_max] = inputs
        return [np.maximum(s_max - ticker, 0)]

    events_table = pa.RecordBatch.from_pylist(events, schema=TS_EVENT_SCHEMA)
    return {
        "events": events_table,
        "expressions": {
            "LOOKBACK": {
                "type": "phrase",
                "inp": [ticker, "MAX_SPOT"],
                "fn": lookback_put_pay_fn,
            },
            "UPDATE": {
                "type": "snapper",
                "inp": [ticker, "MAX_SPOT"],
                "fn": lambda inputs: [np.maximum(inputs[0], inputs[1])],
                "out": ["MAX_SPOT"],
            },
            "INIT": {
                "type": "snapper",
                "inp": [ticker],
                "fn": lambda inputs: inputs,
                "out": ["MAX_SPOT"],
            },
        },
    }


if __name__ == "__main__":
    ticker = "SPX"
    start_date = "2005-09-14"
    T = 0.2
    num_points = 4

    # Creating a look-back put option timetable
    timetable = lookback_put_timetable(ticker, start_date, T, num_points)
    print(timetable["events"].to_pandas())
