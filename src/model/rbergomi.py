"""
Monte Carlo Implementation of the rBergomi model using the scheme proposed by
Mikkel Bennedsen, Asger Lunde, and Mikko S Pakkanen.,
"Hybrid scheme for Brownian semistationary processes.",
Finance and Stochastics, 21(4): 931-965, 2017.
"""
import numpy as np
from numpy.random import SFC64, Generator
from qablet.base.mc import MCModel, MCStateBase
from qablet.base.utils import Forwards


def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a


def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    (Review: is it still optimal when dt is not constant?)
    """
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


class rBergomiMCState(MCStateBase):
    """MCStateClass that implements advance and get_value methods, as needed by the Qablet MCModel interface."""

    def __init__(self, timetable, dataset):
        super().__init__(timetable, dataset)

        # Fetch the common model parameters from the dataset
        self.n = dataset["MC"]["PATHS"]
        self.asset = dataset["rB"]["ASSET"]
        self.asset_fwd = Forwards(dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)

        # Fetch the rBergomi parameters from the dataset
        self.a = dataset["rB"]["ALPHA"]
        self.rho = dataset["rB"]["RHO"]
        self.rho_comp = np.sqrt(1 - self.rho**2)
        self.xi = dataset["rB"]["XI"]
        self.eta = dataset["rB"]["ETA"]

        # Initialize rng, and log stock and variance processes
        self.rng = Generator(SFC64(dataset["MC"]["SEED"]))
        self.x_vec = np.zeros(self.n)  # log stock process
        self.V = np.zeros(self.n)  # variance process
        self.V.fill(self.xi(0))

        # Preallocate arrays for the convolution (for 1000 timesteps)
        self.G = np.zeros(1000)
        self.X = np.zeros((self.n, 1000))
        self.k = 0  # step counter
        self.mean = np.array([0, 0])

        self.cur_time = 0

    def advance(self, new_time):
        """Update x_vec in place when we move simulation by time dt."""

        dt = new_time - self.cur_time
        if dt < 1e-10:
            return

        # generate two sets of random numbers for the hybrid scheme for variance
        dwv = self.rng.multivariate_normal(self.mean, cov(self.a, dt), self.n)

        # and one set for stock, correlated to dwv[:, 0]
        dws = self.rng.normal(0, np.sqrt(dt) * self.rho_comp, self.n)
        dws += self.rho * dwv[:, 0]

        # Update the log stock process first,
        fwd_rate = self.asset_fwd.rate(new_time, self.cur_time)
        self.x_vec += (fwd_rate - self.V / 2.0) * dt
        self.x_vec += np.sqrt(self.V) * dws

        # One part of variance: Riemann sums using convolution
        # Construct arrays for convolution
        self.X[:, self.k] = dwv[:, 0]  # append to X history
        self.k += 1
        if self.k > 1:  # append to G history
            self.G[self.k] = g(b(self.k, self.a) * dt, self.a)
        # Convolution
        Y = np.matmul(self.X[:, : self.k - 1], self.G[self.k : 1 : -1])

        # Add the other part of variance: Exact integrals
        np.add(Y, dwv[:, 1], out=Y)
        # Scale the variance process
        np.multiply(Y, np.sqrt(2 * self.a + 1), out=Y)

        self.V = self.xi(new_time) * np.exp(
            self.eta * Y - 0.5 * self.eta**2 * new_time ** (2 * self.a + 1)
        )

        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the unit at the current time."""

        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)
        else:
            return None


class rBergomiMCModel(MCModel):
    def state_class(self):
        return rBergomiMCState
