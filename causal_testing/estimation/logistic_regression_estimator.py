"""This module contains the LogisticRegressionEstimator class for estimating categorical outcomes."""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_testing.estimation.abstract_regression_estimator import RegressionEstimator
from causal_testing.estimation.effect_estimate import EffectEstimate

logger = logging.getLogger(__name__)


class LogisticRegressionEstimator(RegressionEstimator):
    """A Logistic Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear). It is designed
    for estimating categorical outcomes.
    """

    regressor = sm.Logit

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

    def estimate_unit_odds_ratio(self) -> EffectEstimate:
        """Estimate the odds ratio of increasing the treatment by one. In logistic regression, this corresponds to the
        coefficient of the treatment of interest.

        :return: The odds ratio. Confidence intervals are not yet supported.
        """
        model = self.fit_model(self.df)

        treatment_columns = [
            param
            for param in model.params.index
            if param == self.base_test_case.treatment_variable.name
            or param.startswith(self.base_test_case.treatment_variable.name + "[")
        ]

        confidence_intervals = np.exp(model.conf_int(self.alpha).loc[treatment_columns])

        result = EffectEstimate(
            "unit_odds_ratio",
            pd.Series(np.exp(model.params[treatment_columns])),
            pd.Series(confidence_intervals[0]),
            pd.Series(confidence_intervals[1]),
        )
        return result
