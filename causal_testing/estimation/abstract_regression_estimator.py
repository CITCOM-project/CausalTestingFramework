"""This module contains the RegressionEstimator, which is an abstract class for concrete regression estimators."""

import logging
from typing import Any
from abc import abstractmethod

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper
from patsy import dmatrix  # pylint: disable = no-name-in-module

from causal_testing.specification.variable import Variable
from causal_testing.estimation.abstract_estimator import Estimator

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
        if adjustment_set is None:
            adjustment_set = []
        if formula is not None:
            self.formula = formula
        else:
            terms = [treatment] + sorted(list(adjustment_set)) + sorted(list(effect_modifiers))
            self.formula = f"{outcome} ~ {'+'.join(terms)}"

    @property
    @abstractmethod
    def regressor(self):
        """
        The regressor to use, e.g. ols or logit.
        This should be a property accessible with self.regressor.
        Define as `regressor = ...`` outside of __init__, not as `self.regressor = ...`, otherwise
        you'll get an "cannot instantiate with abstract method" error.
        """

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
        model = self.regressor(formula=self.formula, data=data).fit(disp=0)
        self.model = model
        return model

    def _predict(self, data=None, adjustment_config: dict = None) -> pd.DataFrame:
        """Estimate the outcomes under control and treatment.

        :param data: The data to use, defaults to `self.df`. Controllable for boostrap sampling.
        :param: adjustment_config: The values of the adjustment variables to use.

        :return: The estimated outcome under control and treatment, with confidence intervals in the form of a
                 dataframe with columns "predicted", "se", "ci_lower", and "ci_upper".
        """
        if adjustment_config is None:
            adjustment_config = {}

        model = self._run_regression(data)

        x = pd.DataFrame(columns=self.df.columns)
        x["Intercept"] = 1  # self.intercept
        x[self.treatment] = [self.treatment_value, self.control_value]

        for k, v in adjustment_config.items():
            x[k] = v
        for k, v in self.effect_modifiers.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if str(x.dtypes[col]) == "object":
                x = pd.get_dummies(x, columns=[col], drop_first=True)

        # This has to be here in case the treatment variable is in an I(...) block in the self.formula
        x[self.treatment] = [self.treatment_value, self.control_value]
        return model.get_prediction(x).summary_frame()
