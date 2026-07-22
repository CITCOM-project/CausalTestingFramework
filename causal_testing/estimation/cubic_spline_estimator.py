"""This module contains the CubicSplineRegressionEstimator class, for estimating
continuous outcomes with changes in behaviour"""

import logging
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.specification.variable import Variable
from causal_testing.testing.base_test_case import BaseTestCase

logger = logging.getLogger(__name__)


class CubicSplineRegressionEstimator(LinearRegressionEstimator):
    """A Cubic Spline Regression Estimator is a parametric estimator which restricts the variables in the data to a
    combination of parameters and basis functions of the variables.
    """

    regressor = smf.ols

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        base_test_case: BaseTestCase,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        basis: int,
        effect_modifiers: dict[Variable, Any] = None,
        formula: str = None,
        alpha: float = 0.05,
        expected_relationship=None,
        adjustment_config: dict[Variable, Any] = None,
    ):
        super().__init__(
            base_test_case=base_test_case,
            treatment_value=treatment_value,
            control_value=control_value,
            adjustment_set=adjustment_set,
            effect_modifiers=effect_modifiers,
            formula=formula,
            alpha=alpha,
        )

        self.expected_relationship = expected_relationship
        self.adjustment_config = adjustment_config

        if effect_modifiers is None:
            effect_modifiers = {}

        if formula is None:
            terms = (
                [base_test_case.treatment_variable.name] + sorted(list(adjustment_set)) + sorted(list(effect_modifiers))
            )
            self.formula = f"{base_test_case.outcome_variable.name} ~ cr({'+'.join(terms)}, df={basis})"

    def fit_model(self, df: pd.DataFrame) -> RegressionResultsWrapper:
        """Run linear regression of the treatment and adjustment set against the outcome and return the model.

        :param df: The data to use.
        :return: The model after fitting to data.
        """
        model = self.regressor(formula=self.formula, data=df).fit(disp=0)
        return model

    def estimate_ate_calculated(
        self,
        df: pd.DataFrame,
    ) -> EffectEstimate:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :param df: The data to use.

        :return: The average treatment effect.
        """
        model = self.fit_model(df)

        x = pd.DataFrame({"Intercept": [1], self.base_test_case.treatment_variable.name: [self.treatment_value]})
        if self.adjustment_config is not None:
            for k, v in self.adjustment_config.items():
                x[k] = v
        if self.effect_modifiers is not None:
            for k, v in self.effect_modifiers.items():
                x[k] = v

        treatment = model.predict(x).iloc[0]

        x[self.base_test_case.treatment_variable.name] = self.control_value
        control = model.predict(x).iloc[0]

        return EffectEstimate("ate", pd.Series(treatment - control))
