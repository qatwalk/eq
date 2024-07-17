import numpy as np
import pandas as pd
import pyarrow as pa
from datetime import datetime, timedelta
from qablet_contracts.timetable import py_to_ts
from qablet_contracts.timetable import TS_EVENT_SCHEMA
from qablet.black_scholes.mc import LVMCModel

# Importing the lookback_put_timetable function from lookback.py
from src.contracts.lookback import lookback_put_timetable



MC_PARAMS = {
    "PATHS": 100_000,
    "TIMESTEP": 1 / 250,
    "SEED": 1,
}

# Provided data functions
def basic_info():

    return {
        "prc_dt": datetime(2005, 9, 14),
        "ticker": "SPX",
        "ccy": "USD",
        "spot": 100,  # Using spot price as 100
    }

def assets_data(rate=0.1):
    info = basic_info()

    div_rate = 0
    times = np.array([0.0, 5.0])
    rates = np.array([rate, rate])
    discount_data = ("ZERO_RATES", np.column_stack((times, rates)))

    fwds = info["spot"] * np.exp((rates - div_rate) * times)
    fwd_data = ("FORWARDS", np.column_stack((times, fwds)))

    return {info["ccy"]: discount_data, info["ticker"]: fwd_data}

def localvol_data(rate=0.1):
    info = basic_info()


    return {
        "BASE": "USD",
        "PRICING_TS": py_to_ts(info["prc_dt"]).value,
        "ASSETS": assets_data(rate=rate),
        "MC": MC_PARAMS,
        "LV": {"ASSET": "SPX", "VOL": 0.3},
    }

# Validation of Local Vol for parameters S0=100, r=0.1, T=0.2 Yrs, m=4
ticker = "SPX"
k = 100
spot=100
start_date = '2005-09-14'
T= 0.2
num_points = 4  

# Creating a look-back put option timetable
timetable = lookback_put_timetable(ticker, k,spot, start_date, T, num_points)
print(timetable["events"].to_pandas())

# Pricing with Local Volatility Model
localvol_dataset = localvol_data(rate=0.1)  
localvol_model = LVMCModel()  
price, _ = localvol_model.price(timetable, localvol_dataset)
print(f"LocalVol put price for Table 2 parameters: {price}")

# Validation of Local Vol for parameters S0=100, r=0.1, T=0.5yrs with varying m
m_values = [5, 10, 20, 40, 80, 160]  
T= 0.5

for m in m_values:
    timetable = lookback_put_timetable(ticker, k, spot, start_date, T, m)
    print(f"\nTimetable for m={m}:\n", timetable["events"].to_pandas())

    # Pricing with Local Volatility Model
    price, _ = localvol_model.price(timetable, localvol_dataset)
    print(f"LocalVol put price for m={m}: {price}")
