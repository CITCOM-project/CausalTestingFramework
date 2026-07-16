"""
This module contains code to measure various aspects of causal test adequacy.
"""

import logging

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
        kurtosis=None,
        passing=None,
        results=None,
        successful=None,
    ):
        self.kurtosis = kurtosis
        self.passing = passing
        self.results = results
        self.successful = successful

    def to_dict(self):
        """Returns the adequacy object as a dictionary."""
        return {
            "kurtosis": self.kurtosis.to_dict(),
            "passing": self.passing,
            "successful": self.successful,
            "results": self.results.reset_index(drop=True).to_dict(),
        }
