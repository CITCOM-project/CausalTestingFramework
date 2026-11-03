"""
This module contains code to measure various aspects of causal test adequacy.
"""

import logging

import numpy as np
from pandas import Series

logger = logging.getLogger(__name__)


class DataAdequacy:
    """
    Measures the adequacy of a given test according to the Fisher kurtosis of the bootstrapped result.

    * Positive kurtoses indicate the model doesn't have enough data so is unstable.
    * Negative kurtoses indicate the model doesn't have enough data, but is too stable,
      indicating that the spread of inputs is insufficient.
    * Zero kurtosis is optimal.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        kurtosis: Series = None,
        passing: int = None,
        results: dict = None,
        successful: int = None,
        bootstrap_size: int = None,
    ):
        self.kurtosis = Series(kurtosis) if kurtosis is not None else None
        self.passing = passing
        self.results = results
        self.successful = successful
        self.bootstrap_size = bootstrap_size

    def to_dict(self, include_adequacy_results: bool = False):
        """
        :returns: the adequacy object as a dictionary.
        :param include_adequacy_results: Whether to serialise the results.
        """
        result = {
            "kurtosis": {str(k): v for k, v in self.kurtosis.to_dict().items()} if self.kurtosis is not None else None,
            "kurtosis_simplified": (
                {str(k): v for k, v in self.kurtosis_simplified().to_dict().items()}
                if self.kurtosis is not None
                else None
            ),
            "passing": self.passing,
            "successful": self.successful,
            "bootstrap_size": self.bootstrap_size,
        }
        if include_adequacy_results:
            return result | {"results": self.results.reset_index(drop=True).to_dict()}
        return result

    def kurtosis_simplified(self, halfway: float = -0.8):
        """
        Converts raw kurtosis value to a 0-100% score for easier interpretation.
        Kurtosis value 0 represents 100%.
        As the absolute value of kurtosis moves further away from 0, the converted value decreases.

        :param halfway: The absolute value of kurtosis that maps to 50%. (Defaults to -0.8)
            Since a uniform distribution has kurtosis value -1.2, which represents pretty much the worst case that it's
            possible to achieve in this setting, a kurtosis of -0.8 corresponds to a curve that is visually about
            "halfway" between a normal distribution and a uniform one.
        """
        # Scaling factor so that |x| = halfway yields exactly 50%
        scaling_factor = np.log(3) / abs(halfway)  # ~1.0986 / threshold

        # Sigmoid decay function: 2 / (1 + exp(k * |x|))
        score = 2.0 / (1.0 + np.exp(scaling_factor * np.abs(self.kurtosis)))

        return score * 100.0
