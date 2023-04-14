"""This module contains the class EnumGen, which allows us to easily create generating uniform distributions from enums."""

from scipy.stats import rv_discrete
from enum import Enum
import numpy as np


class EnumGen(rv_discrete):
    """This class allows us to easily create generating uniform distributions from enums. This is helpful for generating concrete test inputs from abstract test cases."""

    def __init__(self, datatype: Enum):
        self.dt = dict(enumerate(datatype, 1))
        self.inverse_dt = {v: k for k, v in self.dt.items()}

    def ppf(self, q):
        """Percent point function (inverse of `cdf`) at q of the given RV.
        Parameters
        ----------
        q : array_like
            Lower tail probability.
        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.
        """
        return np.vectorize(self.dt.get)(np.ceil(len(self.dt) * q))

    def cdf(self, q):
        """
        Cumulative distribution function of the given RV.
        Parameters
        ----------
        q : array_like
            quantiles
        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`
        """
        return np.vectorize(self.inverse_dt.get)(q) / len(self.dt)
