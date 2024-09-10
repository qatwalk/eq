"""
Monte Carlo Implementation of a local vol model using finmc interface.
The model takes volatility as SVI parameters and calculates local volatility using Dupire's formula.
"""

from math import sqrt

import numpy as np
from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards
from finmc.utils.bs import dupire_local_var
from finmc.utils.interp import UniformGridInterp
from finmc.utils.mc import antithetic_normal
from numpy.random import SFC64, Generator


def svi_tvar(params, k):
    """Get total variance (v^2 t) from SVI params and log strikes."""
    k_m = k - params["m"]
    discr = np.sqrt(k_m**2 + params["sig"] ** 2)
    w = params["a"] + params["b"] * (params["rho"] * k_m + discr)
    return w


def svi_vol(params, k):
    """Get vol from SVI params and log strikes."""
    w = svi_tvar(params, k)  # total variance
    return np.sqrt(w / params["texp"])


def interp_vec(t, t_vec, w_vec):
    """Given a 2-D array w_vec[t,k], return the interpolated 1D array w[k] at given time t."""

    i_left = np.searchsorted(t_vec, t, side="left")
    i_right = np.searchsorted(t_vec, t, side="right")

    if i_left == i_right:
        if i_right == 0:
            w = w_vec[0] * (t / t_vec[0])
        elif i_right == len(t_vec):
            w = w_vec[-1]  # Consider extrapolation using the last slope
        else:
            i_left -= 1
            t_right = t_vec[i_right]
            t_left = t_vec[i_left]
            den = t_right - t_left
            w = (
                w_vec[i_left]
                + (w_vec[i_right] - w_vec[i_left]) * (t - t_left) / den
            )
    else:
        w = w_vec[i_left]
    return w


class SVItoLV:
    """Helper class to convert SVI parameters to local volatility."""

    def __init__(self, svi_df, shape):
        self.ugi = UniformGridInterp()  # fast interpolator for uniform x-grid

        # strike grid with one more point on each side
        self.k_vec = np.pad(self.ugi.x_vec, 1)
        self.dk = self.ugi.dx
        self.k_vec[0] = self.k_vec[1] - self.dk
        self.k_vec[-1] = self.k_vec[-2] + self.dk

        self.t_vec = svi_df["texp"]

        # Create a 2-D array of total variances (ws) by t and k.
        self.ws = np.zeros((len(self.t_vec), len(self.k_vec)))
        for i, t in enumerate(self.t_vec):
            self.ws[i] = svi_tvar(svi_df.loc[i], self.k_vec)

        self.w_vec_prev = np.zeros(len(self.k_vec))  # total variance at t0
        self.vol = np.zeros(shape)  # pre-allocate array for vol by path

    def advance_vol(self, prev_time, new_time, x_vec):
        """Advance w_vec to new_time and stores the local volatility between prev_time and new time.
        Returns the local volatility for the given x_vec."""

        # get total variance for the strike grid,  at new_time
        w_vec = interp_vec(new_time, self.t_vec, self.ws)
        # get local variance for the strike grid,  between prev_time and new_time
        lvar = dupire_local_var(
            new_time - prev_time,
            self.dk,
            self.k_vec[1:-1],
            self.w_vec_prev,
            w_vec,
        )
        self.w_vec_prev = w_vec

        # get local vol by path
        self.ugi.interp(x_vec, np.sqrt(lvar), out=self.vol)


# Define a class for the state of a single asset BS Local Vol MC process
class LVMC(MCFixedStep):
    def reset(self):
        # fetch the model parameters from the dataset
        self.n = self.dataset["MC"]["PATHS"]
        self.timestep = self.dataset["MC"]["TIMESTEP"]

        self.asset = self.dataset["LV"]["ASSET"]
        self.asset_fwd = Forwards(self.dataset["ASSETS"][self.asset])
        self.discounter = Discounter(
            self.dataset["ASSETS"][self.dataset["BASE"]]
        )

        # initialize helper for local vol calibration
        self.localvol = SVItoLV(self.dataset["LV"]["VOL"], self.n)

        # Initialize rng and any arrays
        self.rng = Generator(SFC64(self.dataset["MC"]["SEED"]))
        self.x_vec = np.zeros(self.n)  # process x (log stock)
        self.dz_vec = np.empty(self.n, dtype=np.float64)
        self.tmp = np.empty(self.n, dtype=np.float64)

        self.cur_time = 0

    def step(self, new_time):
        """Update x_vec in place when we move simulation by time dt."""

        dt = new_time - self.cur_time

        # update local vol, from current time to new time
        self.localvol.advance_vol(self.cur_time, new_time, self.x_vec)

        # generate the random numbers and advance the log stock process
        antithetic_normal(self.rng, self.n, sqrt(dt), self.dz_vec)
        self.dz_vec *= self.localvol.vol

        # add drift to x_vec: - vol * vol * dt / 2.0
        np.multiply(self.localvol.vol, self.localvol.vol, out=self.tmp)
        self.tmp *= -0.5 * dt
        self.x_vec += self.tmp
        # add the random part to x_vec
        self.x_vec += self.dz_vec

        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the modeled asset at the current time.
        otherwise return none."""

        if unit == self.asset:
            return self.asset_fwd.forward(self.cur_time) * np.exp(self.x_vec)

    def get_df(self):
        return self.discounter.discount(self.cur_time)
