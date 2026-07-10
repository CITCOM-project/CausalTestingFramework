"""This module contains the LogisticRegressionEstimator class for estimating categorical outcomes."""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_testing.estimation.abstract_regression_estimator import RegressionEstimator
from causal_testing.estimation.effect_estimate import EffectEstimate

logger = logging.getLogger(__name__)


class MultinomialRegressionEstimator(RegressionEstimator):
    """A Logistic Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear). It is designed
    for estimating categorical outcomes.
    """

    regressor = sm.MNLogit

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        super().add_modelling_assumptions()
        self.modelling_assumptions.append("Outcome is categorical.")

    def estimate_unit_odds_ratio(self) -> EffectEstimate:
        """Estimate the odds ratio of increasing the treatment by one. In logistic regression, this corresponds to the
        coefficient of the treatment of interest.

        :return: The odds ratio with confidence intervals.
        """
        model = self.fit_model(self.df)

        conf_int = model.conf_int(self.alpha)
        levels_of_interest = [
            (level, covariate) for level, covariate in conf_int.index if covariate in self.treatment_columns(model)
        ]
        confidence_intervals = np.exp(conf_int.loc[levels_of_interest])

        # Format the params to a MultiIndexed Series like the confidence intervals for consistent indexing
        stacked_params = model.params.stack(dropna=False)
        multi_indexed_params = stacked_params.swaplevel(0, 1).sort_index()
        multi_indexed_params.index = conf_int.index

        result = EffectEstimate(
            "unit_odds_ratio",
            pd.Series(np.exp(multi_indexed_params[levels_of_interest])),
            pd.Series(confidence_intervals["lower"]),
            pd.Series(confidence_intervals["upper"]),
        )
        return result
