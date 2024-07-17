import numpy as np
import pandas as pd
import pyarrow as pa
from datetime import datetime, timedelta
from qablet_contracts.timetable import py_to_ts
from qablet_contracts.timetable import TS_EVENT_SCHEMA
from qablet.black_scholes.mc import LVMCModel

# Creating a look-back put option timetable with custom lookbacks
def lookback_put_timetable(ticker, k, spot, start_date, maturity, num_points):
    
    start_date = pd.to_datetime(start_date)
    maturity = pd.to_datetime(maturity)
    
    fix_dates = pd.date_range(start=start_date, end=maturity, periods=num_points)
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
        [s_min] = inputs
        return [np.maximum(k - s_min, 0)]
    
    events_table = pa.RecordBatch.from_pylist(events, schema=TS_EVENT_SCHEMA)
    return {
        "events": events_table,
        "expressions": {
            "LOOKBACK": {
                "type": "phrase",
                "inp": ["MIN_PRICE"],
                "fn": lookback_put_pay_fn,
            },
            "UPDATE": {
                "type": "snapper",
                "inp": [ticker, "MIN_PRICE"],
                "fn": lambda inputs: [np.minimum(inputs[0], inputs[1])],
                "out": ["MIN_PRICE"],
            },
            "INIT": {
                "type": "snapper",
                "inp": [],
                "fn": lambda inputs: [spot],
                "out": ["MIN_PRICE"],
            },
        },
    }
