"""
Monte Carlo Implementation of local vol model. The model takes volatility as
SVI parameters and calculates local volatility using Dupire's formula.
"""

from math import sqrt

import numpy as np
from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards
from numpy.random import SFC64, Generator


def svi(params, k):
    """Get total variance (v^2 t) from SVI params and log strikes."""
    k_m = k - params["m"]
    discr = np.sqrt(k_m**2 + params["sig"] ** 2)
    w = params["a"] + params["b"] * (params["rho"] * k_m + discr)
    return w


def dupire_local_var(dt, dk, k, w_vec_prev, w_vec):
    """Calculate local variance (vol**2) from t0 to t1, given a list of strikes k, uniformly spaced by dk,
    and the total variances (w = T * vol**2) at time t0 and t1.
    The result vector is two elements shorter than the input vectors k and w."""

    w = (w_vec + w_vec_prev) / 2
    # time derivative of w
    wt = (w_vec - w_vec_prev) / dt

    # strike derivatives of w
    wk = (w[2:] - w[:-2]) / (2 * dk)
    wkk = (w[2:] + w[:-2] - 2 * w[1:-1]) / (dk**2)

    # drop the extra points at top and bottom
    w_ = w[1:-1]
    wt_ = wt[1:-1]

    # apply Dupire's formula
    return wt_ / (
        1
        - k / w_ * wk
        + 1 / 4 * (-1 / 4 - 1 / w_ + k**2 / w_**2) * (wk) ** 2
        + 1 / 2 * wkk
    )


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


class UniformGridInterp:
    """Helper class to interpolate, when x-array is uniformly spaced.
    This avoids the cost of index search in arbitrary x-array."""

    def __init__(self, xmin=-3.0, xmax=3.0, dx=0.01):
        """Initialize the x-range and allocate some arrays."""

        self.dx = dx
        self.xmin = xmin
        self.x_vec = np.arange(xmin, xmax + dx / 2, dx)
        self.xlen = len(self.x_vec)

        # Pre-allocate arrays
        self.slope = np.zeros(self.xlen)

    def interp(self, x_vec, y_vec, out):
        """Interpolate y_vec at x_vec and store the result in out."""

        # Find the left index of the interval containing x: idx = floor((x - xmin) / dx)
        # Reusing the out array as a temp array, as it is the same shape
        np.subtract(x_vec, self.xmin, out=out)
        np.divide(out, self.dx, out=out)
        idx = out.astype(int)

        # Clip the index to [0, xlen - 1]
        np.clip(idx, 0, self.xlen - 1, out=idx)

        # get slope of y in each interval
        np.subtract(y_vec[1:], y_vec[:-1], out=self.slope[:-1])
        self.slope[-1] = self.slope[-2]
        np.divide(self.slope, self.dx, out=self.slope)

        # y = (x - left) * slope + left
        np.subtract(x_vec, self.x_vec[idx], out=out)  # x - left
        np.multiply(out, self.slope[idx], out=out)  # * slope
        np.add(out, y_vec[idx], out=out)  # + left


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
            self.ws[i] = svi(svi_df.loc[i], self.k_vec)

        self.w_vec_prev = np.zeros(len(self.k_vec))  # total variance at t0
        self.vol = np.zeros(shape)  # pre-allocate array for vol by path

    def advance_vol(self, prev_time, new_time, x_vec):
        """Advance w_vec to new_time and stores the local volatility between prev_time and new time.
        Returns the local volatility for the given x_vec."""
        w_vec = interp_vec(new_time, self.t_vec, self.ws)
        lvar = dupire_local_var(
            new_time - prev_time,
            self.dk,
            self.k_vec[1:-1],
            self.w_vec_prev,
            w_vec,
        )

        self.w_vec_prev = w_vec
        lvol = np.sqrt(lvar)  # vol by strike grid
        self.ugi.interp(x_vec, lvol, out=self.vol)  # get vol by path


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
        self.rng.standard_normal(self.n, out=self.dz_vec)
        self.dz_vec *= sqrt(dt)
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
