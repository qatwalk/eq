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


class SVItoLV:
    """Helper class to convert SVI parameters to local volatility."""

    def __init__(self, svi_df, shape):
        """Create a 2-D array of total variances (ws) by t and k.
        The knots in t are the same as in the input svi_df."""
        kmin, kmax, dk = -4.0, 4.0, 0.02
        self.dk = dk
        self.kmin = kmin
        self.kmax = kmax
        self.k_vec_plus = np.arange(kmin - dk, kmax + dk + dk / 2, dk)
        self.k_vec = self.k_vec_plus[1:-1]
        self.klen = len(self.k_vec)
        self.slope = np.zeros(self.klen)
        self.vol = np.zeros(shape)  # vol for each path
        self.idx = np.zeros(
            shape, dtype=int
        )  # strike grid index for each path
        self.tmp = np.zeros(shape)  # temporary array for interpolation

        self.t_vec = svi_df["texp"]
        self.t_len = len(self.t_vec)

        self.ws = np.zeros((len(self.t_vec), len(self.k_vec_plus)))
        for i, t in enumerate(self.t_vec):
            self.ws[i] = svi(svi_df.loc[i], self.k_vec_plus)

        self.w_vec_prev = np.zeros(len(self.k_vec_plus))

    def total_var(self, t):
        """Return the total variance (v^2 t) vector at time t
        by interpolating from self.ws."""

        i_left = np.searchsorted(self.t_vec, t, side="left")
        i_right = np.searchsorted(self.t_vec, t, side="right")

        if i_left == i_right:
            if i_right == 0:
                w = self.ws[0] * (t / self.t_vec[0])
            elif i_right == self.t_len:
                w = self.ws[-1]  # Consider extrapolation using the last slope
            else:
                i_left -= 1
                t_right = self.t_vec[i_right]
                t_left = self.t_vec[i_left]
                den = t_right - t_left
                w = (
                    self.ws[i_left]
                    + (self.ws[i_right] - self.ws[i_left]) * (t - t_left) / den
                )
        else:
            w = self.ws[i_left]
        return w

    def advance_vol(self, prev_time, new_time, x_vec):
        """Advance w_vec to new_time and stores the local volatility between prev_time and new time.
        Returns the local volatility for the given x_vec."""
        w_vec = self.total_var(new_time)
        lvar = self._svi_local_var_step(
            prev_time,
            new_time,
            self.w_vec_prev,
            w_vec,
        )

        self.w_vec_prev = w_vec
        self.lvol = np.sqrt(lvar)  # vol for each grid point
        self._k_interp(x_vec)  # interpolate vol for each path

    def _k_interp(self, x_vec):
        # Find the left index of the interval containing x
        np.subtract(x_vec, self.kmin, out=self.tmp)
        np.divide(self.tmp, self.dk, out=self.tmp)

        # idxf = (x_vec - self.kmin) / self.dk
        self.idx = self.tmp.astype(int)
        np.clip(self.idx, 0, self.klen - 1, out=self.idx)
        x_base = self.k_vec[self.idx]

        # get slope of lvar from one strike to the next
        np.subtract(self.lvol[1:], self.lvol[:-1], out=self.slope[:-1])
        self.slope[-1] = self.slope[-2]
        np.divide(self.slope, self.dk, out=self.slope)

        np.subtract(x_vec, x_base, out=self.vol)  # xfrac
        np.multiply(self.vol, self.slope[self.idx], out=self.vol)  # yfrac
        np.add(self.vol, self.lvol[self.idx], out=self.vol)  # add the base

    def _svi_local_var_step(self, t0, t1, w_vec_prev, w_vec):
        """Calculate local variance (vol**2) from t0 to t1, given a list of strikes k, uniformly spaced by dk,
        and the total variances (w = T * vol**2) at time t0 and t1.
        The result vector is two elements shorter than the input vectors k and w."""

        w = (w_vec + w_vec_prev) / 2
        # time derivative of w
        wt = (w_vec - w_vec_prev) / (t1 - t0)

        # strike derivatives of w
        wk = (w[2:] - w[:-2]) / (2 * self.dk)
        wkk = (w[2:] + w[:-2] - 2 * w[1:-1]) / (self.dk**2)

        # drop the extra points at top and bottom
        w_ = w[1:-1]
        k_ = self.k_vec_plus[1:-1]
        wt_ = wt[1:-1]

        # apply Dupire's formula
        return wt_ / (
            1
            - k_ / w_ * wk
            + 1 / 4 * (-1 / 4 - 1 / w_ + k_**2 / w_**2) * (wk) ** 2
            + 1 / 2 * wkk
        )


# Define a class for the state of a single asset BS Local Vol MC process
class LVMC(MCFixedStep):
    def reset(self):
        # fetch the model parameters from the dataset
        self.n = self.dataset["MC"]["PATHS"]
        self.timestep = self.dataset["MC"]["TIMESTEP"]

        self.asset = self.dataset["LV"]["ASSET"]
        self.asset_fwd = Forwards(self.dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)
        self.discounter = Discounter(
            self.dataset["ASSETS"][self.dataset["BASE"]]
        )

        # initialize helper for local vol calibration
        self.localvol = SVItoLV(self.dataset["LV"]["VOL"], self.n)
        self.logspot = np.log(self.spot)

        # Initialize rng and any arrays
        self.rng = Generator(SFC64(self.dataset["MC"]["SEED"]))
        self.x_vec = np.zeros(self.n)  # process x (log stock)
        self.dz_vec = np.empty(self.n, dtype=np.float64)
        self.tmp = np.empty(self.n, dtype=np.float64)

        self.cur_time = 0

    def step(self, new_time):
        """Update x_vec in place when we move simulation by time dt."""

        dt = new_time - self.cur_time
        fwd_rate = self.asset_fwd.rate(new_time, self.cur_time)

        # Get local vol, from current time to new time)
        logfwd_shift = self.asset_fwd.log_forward_fn(new_time) - self.logspot
        np.subtract(self.x_vec, logfwd_shift, out=self.tmp)
        self.localvol.advance_vol(self.cur_time, new_time, self.tmp)
        # Todo: should I use new x_vec or old?
        # should I use new log shift or old?

        # # generate the random numbers and advance the log stock process
        self.rng.standard_normal(self.n, out=self.dz_vec)
        self.dz_vec *= sqrt(dt)
        self.dz_vec *= self.localvol.vol

        # add drift to x_vec: (fwd_rate - vol * vol / 2.0) * dt
        np.multiply(self.localvol.vol, self.localvol.vol, out=self.tmp)
        self.tmp *= -0.5 * dt
        self.tmp += fwd_rate * dt
        self.x_vec += self.tmp
        # add the random part to x_vec
        self.x_vec += self.dz_vec

        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the modeled asset at the current time.
        otherwise return none."""

        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)

    def get_df(self):
        return self.discounter.discount(self.cur_time)
