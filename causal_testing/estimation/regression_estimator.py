"""This module contains the RegressionEstimator, which is an abstract class for concrete regression estimators."""

import logging
from typing import Any
from abc import abstractmethod, abstractmethod

import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix  # pylint: disable = no-name-in-module
from patsy import ModelDesc
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_testing.specification.variable import Variable
from causal_testing.estimation.gp import GP
from causal_testing.estimation.estimator import Estimator

logger = logging.getLogger(__name__)


class RegressionEstimator(Estimator):
    """A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        df: pd.DataFrame = None,
        effect_modifiers: dict[Variable:Any] = None,
        formula: str = None,
        alpha: float = 0.05,
        query: str = "",
    ):
        super().__init__(
            treatment=treatment,
            treatment_value=treatment_value,
            control_value=control_value,
            adjustment_set=adjustment_set,
            outcome=outcome,
            df=df,
            effect_modifiers=effect_modifiers,
            query=query,
        )

        self.model = None
        if effect_modifiers is None:
            effect_modifiers = []
        if formula is not None:
            self.formula = formula
        else:
            terms = [treatment] + sorted(list(adjustment_set)) + sorted(list(effect_modifiers))
            self.formula = f"{outcome} ~ {'+'.join(terms)}"
        for term in self.effect_modifiers:
            self.adjustment_set.add(term)

    @property
    @abstractmethod
    def regression(self):
        raise NotImplementedError("Subclasses must implement the 'model' property.")

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

    def _run_regression(self, data=None) -> RegressionResultsWrapper:
        """Run logistic regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        if data is None:
            data = self.df
        model = self.regression(formula=self.formula, data=data).fit(disp=0)
        self.model = model
        return model
