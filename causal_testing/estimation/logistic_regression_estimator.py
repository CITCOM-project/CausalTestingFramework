"""This module contains the LogisticRegressionEstimator class for estimating categorical outcomes."""

import logging

import numpy as np
import statsmodels.formula.api as smf

from causal_testing.estimation.abstract_regression_estimator import RegressionEstimator

logger = logging.getLogger(__name__)


class LogisticRegressionEstimator(RegressionEstimator):
    """A Logistic Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear). It is designed
    for estimating categorical outcomes.
    """

    regressor = smf.logit

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions.append(
            "The variables in the data must fit a shape which can be expressed as a linear"
            "combination of parameters and functions of variables. Note that these functions"
            "do not need to be linear."
        )
        self.modelling_assumptions.append("The outcome must be binary.")
        self.modelling_assumptions.append("Independently and identically distributed errors.")

    def estimate_unit_odds_ratio(self) -> float:
        """Estimate the odds ratio of increasing the treatment by one. In logistic regression, this corresponds to the
        coefficient of the treatment of interest.

        :return: The odds ratio. Confidence intervals are not yet supported.
        """
        model = self._run_regression(self.df)
        return np.exp(model.params[self.treatment])
