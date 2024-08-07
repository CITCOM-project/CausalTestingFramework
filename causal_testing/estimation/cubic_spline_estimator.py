"""This module contains the CubicSplineRegressionEstimator class, for estimating
continuous outcomes with changes in behaviour"""

import logging
from typing import Any

import pandas as pd

from causal_testing.specification.variable import Variable
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator

logger = logging.getLogger(__name__)


class CubicSplineRegressionEstimator(LinearRegressionEstimator):
    """A Cubic Spline Regression Estimator is a parametric estimator which restricts the variables in the data to a
    combination of parameters and basis functions of the variables.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        basis: int,
        df: pd.DataFrame = None,
        effect_modifiers: dict[Variable:Any] = None,
        formula: str = None,
        alpha: float = 0.05,
        expected_relationship=None,
    ):
        super().__init__(
            treatment, treatment_value, control_value, adjustment_set, outcome, df, effect_modifiers, formula, alpha
        )

        self.expected_relationship = expected_relationship

        if effect_modifiers is None:
            effect_modifiers = []

        if formula is None:
            terms = [treatment] + sorted(list(adjustment_set)) + sorted(list(effect_modifiers))
            self.formula = f"{outcome} ~ cr({'+'.join(terms)}, df={basis})"

    def estimate_ate_calculated(self, adjustment_config: dict = None) -> pd.Series:
        model = self._run_linear_regression()

        x = {"Intercept": 1, self.treatment: self.treatment_value}
        if adjustment_config is not None:
            for k, v in adjustment_config.items():
                x[k] = v
        if self.effect_modifiers is not None:
            for k, v in self.effect_modifiers.items():
                x[k] = v

        treatment = model.predict(x).iloc[0]

        x[self.treatment] = self.control_value
        control = model.predict(x).iloc[0]

        return pd.Series(treatment - control)
