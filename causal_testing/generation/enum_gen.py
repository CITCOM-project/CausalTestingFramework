"""This module contains the class EnumGen, which allows us to easily create
generating uniform distributions from enums."""

from enum import Enum
from scipy.stats import rv_discrete
import numpy as np


class EnumGen(rv_discrete):
    """This class allows us to easily create generating uniform distributions
    from enums. This is helpful for generating concrete test inputs from
    abstract test cases."""

    def __init__(self, datatype: Enum):
        super().__init__()
        self.datatype = dict(enumerate(datatype, 1))
        self.inverse_dt = {v: k for k, v in self.datatype.items()}

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
        return np.vectorize(self.datatype.get)(np.ceil(len(self.datatype) * q))

    def cdf(self, k):
        """
        Cumulative distribution function of the given RV.
        Parameters
        ----------
        k : array_like
            quantiles
        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`
        """
        return np.vectorize(self.inverse_dt.get)(k) / len(self.datatype)
