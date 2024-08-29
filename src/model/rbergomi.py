"""
Monte Carlo Implementation of the rBergomi model using Qablet MCModel Interface.
Uses the scheme proposed by Mikkel Bennedsen, Asger Lunde, and Mikko S Pakkanen.,
"Hybrid scheme for Brownian semistationary processes.",
Finance and Stochastics, 21(4): 931-965, 2017.
"""

import numpy as np
from finmc.models.base import MCBase
from finmc.utils.assets import Discounter, Forwards
from numpy.random import SFC64, Generator


def g(x, a):
    """TBSS kernel applicable to the rBergomi variance process."""
    return x**a


def b(k, a):
    """Optimal discretisation of TBSS process for minimising hybrid scheme error."""
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov(a, dt):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for tractability.
    """
    cov = np.array([[0.0, 0.0], [0.0, 0.0]])
    cov[0, 0] = dt
    cov[0, 1] = (dt ** (1.0 * a + 1)) / (1.0 * a + 1)
    cov[1, 1] = (dt ** (2.0 * a + 1)) / (2.0 * a + 1)
    cov[1, 0] = cov[0, 1]
    return cov


class rBergomiMC(MCBase):
    """MCStateClass that implements advance and get_value methods, as needed by the Qablet MCModel interface."""

    def reset(self):
        """Fetch any information from the dataset or timetable, that needs to be stored into self,
        to facilitate the 'advance' method to run as quickly as possible."""

        # Fetch the common model parameters from the dataset
        self.n = self.dataset["MC"]["PATHS"]
        self.asset = self.dataset["rB"]["ASSET"]
        self.asset_fwd = Forwards(self.dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)
        self.discounter = Discounter(
            self.dataset["ASSETS"][self.dataset["BASE"]]
        )

        # Fetch the rBergomi parameters from the dataset
        self.a = self.dataset["rB"]["ALPHA"]
        self.rho = self.dataset["rB"]["RHO"]
        self.rho_comp = np.sqrt(1 - self.rho**2)
        self.xi = self.dataset["rB"]["XI"]
        self.eta = self.dataset["rB"]["ETA"]

        self.dt = 1 / 250  # vol time step, currently hardcoded.

        # Initialize rng, and log stock and variance processes
        self.rng = Generator(SFC64(self.dataset["MC"]["SEED"]))
        self.V = np.zeros(self.n)  # variance process
        self.V.fill(self.xi(0))

        # Preallocate arrays for the convolution (for 1000 timesteps)
        self.G = np.zeros(1000)
        self.X = np.zeros((self.n, 1000))
        self.k = 0  # step counter
        self.mean = np.array([0, 0])
        self.cov = cov(self.a, self.dt)

        self.v_time = 0  # when V was updated
        self.x_time = 0  # when x_vec was updated, always on or after v_time
        self.x_vec = np.zeros(self.n)  # log stock process

        self.gen_rands()

    def gen_rands(self):
        """Create random numbers for the next time step."""
        # two sets for the hybrid scheme for variance
        self.dwv = self.rng.multivariate_normal(self.mean, self.cov, self.n)
        # one set for stock, correlated to dwv[:, 0]
        dws = self.rng.normal(0, np.sqrt(self.dt) * self.rho_comp, self.n)
        dws += self.rho * self.dwv[:, 0]
        self.x_diff = np.sqrt(self.V) * dws - self.V / 2.0 * self.dt

    def advance_x_vec(self, new_time):
        """Update the log stock process."""
        fwd_rate = self.asset_fwd.rate(new_time, self.x_time)
        self.x_vec += fwd_rate * (new_time - self.x_time)
        self.x_vec += self.x_diff * (new_time - self.x_time) / self.dt
        self.x_time = new_time

    def advance(self, new_time):
        """Update x_vec and V in place when we move simulation by time dt."""

        while new_time > self.v_time + self.dt:
            # update both x_vec and V by one timestep (dt)
            new_v_time = self.v_time + self.dt

            # Update the log stock process
            self.advance_x_vec(new_v_time)

            # One part of variance: Riemann sums using convolution
            self.X[:, self.k] = self.dwv[:, 0]  # append to X history

            if self.k > 0:  # append to G history
                self.G[self.k + 1] = g(b(self.k + 1, self.a) * self.dt, self.a)

            # Convolution
            Y = np.matmul(self.X[:, : self.k], self.G[self.k + 1 : 1 : -1])
            self.k += 1

            # Add the other part of variance: Exact integrals
            np.add(Y, self.dwv[:, 1], out=Y)
            # Scale the variance process
            np.multiply(Y, np.sqrt(2 * self.a + 1), out=Y)

            self.V = self.xi(new_v_time) * np.exp(
                self.eta * Y
                - 0.5 * self.eta**2 * new_v_time ** (2 * self.a + 1)
            )

            self.gen_rands()
            self.v_time = new_v_time

        if new_time > self.x_time + 1e-10:
            self.advance_x_vec(new_time)

    def get_value(self, unit):
        """Return the value of the unit at the current time."""

        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)

    def get_df(self):
        return self.discounter.discount(self.cur_time)
