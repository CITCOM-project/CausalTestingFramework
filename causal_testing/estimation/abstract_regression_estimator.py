"""This module contains the RegressionEstimator, which is an abstract class for concrete regression estimators."""

import logging
from abc import abstractmethod
from typing import Any

import pandas as pd
from patsy import dmatrices, dmatrix  # pylint: disable = no-name-in-module
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.specification.variable import Variable
from causal_testing.testing.base_test_case import BaseTestCase

logger = logging.getLogger(__name__)


class RegressionEstimator(Estimator):
    """A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        base_test_case: BaseTestCase,
        control_value: float = None,
        treatment_value: float = None,
        adjustment_set: set = None,
        effect_modifiers: dict[Variable, Any] = None,
        adjustment_config: dict[Variable, Any] = None,
        formula: str = None,
        alpha: float = 0.05,
    ):
        # pylint: disable=R0801
        super().__init__(
            base_test_case=base_test_case,
            control_value=control_value,
            treatment_value=treatment_value,
            adjustment_set=adjustment_set,
            effect_modifiers=effect_modifiers,
            alpha=alpha,
        )

        if effect_modifiers is None:
            effect_modifiers = {}
        self.adjustment_config = {} if adjustment_config is None else adjustment_config
        if adjustment_set is None:
            adjustment_set = []
        if formula is not None:
            self.formula = formula
        else:
            terms = (
                [base_test_case.treatment_variable.name] + sorted(list(adjustment_set)) + sorted(list(effect_modifiers))
            )
            self.formula = f"{base_test_case.outcome_variable.name} ~ {'+'.join(terms)}"

        for term in list(self.effect_modifiers) + list(self.adjustment_config):
            self.adjustment_set.add(term)

    def _setup_covariates(self, df: pd.DataFrame) -> pd.Series:
        """
        Parse the formula and set up the covariates from the design matrix so we can use them in the statsmodels array
        API. This allows us to only parse the formula once, rather than using the formula API, which parses it every
        time the regression model is fit, which can be a lot if using causal test adequacy.
        :param df: The data to use.
        :returns: The data and the covariate columns.
        """
        _, covariate_data = dmatrices(self.formula, df, return_type="dataframe")
        df = pd.concat([df, covariate_data[[col for col in covariate_data.columns if col not in df]]], axis=1)
        covariates = covariate_data.columns.tolist()
        return covariates, df.dropna(subset=covariates)

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

    def fit_model(self, df: pd.DataFrame) -> RegressionResultsWrapper:
        """Run logistic regression of the treatment and adjustment set against the outcome and return the model.

        :param df: The data to use.
        :return: The model after fitting to data.
        """
        covariates, df = self._setup_covariates(df)
        model = self.regressor(df[self.base_test_case.outcome_variable.name], df[covariates]).fit(disp=0)
        return model

    def treatment_columns(self, model: RegressionResultsWrapper) -> list[str]:
        """
        Get the names of the treatment columns from the model.
        This is a workaround for statsmodels mangling the names of categorical variables to include the values.

        :param model: The fitted model from which to extract the variable names.
        :returns: A list of the feature names in the model that represent the treatment. Normally this will just be
        [treatment_name], but for categorical treatments, you'll have
        [treatment_name[value_1], treatment_name[value_2]].
        """
        return [
            param
            for param in model.params.index
            if param == self.base_test_case.treatment_variable.name
            or param.startswith(self.base_test_case.treatment_variable.name + "[")
        ]

    def _predict(self, df) -> pd.DataFrame:
        """Estimate the outcomes under control and treatment.

        :param df: The data to use.
        :param: adjustment_config: The values of the adjustment variables to use.

        :return: The estimated outcome under control and treatment, with confidence intervals in the form of a
                 dataframe with columns "predicted", "se", "ci_lower", and "ci_upper".
        """
        model = self.fit_model(df)

        x = pd.DataFrame(columns=df.columns)
        x["Intercept"] = 1  # self.intercept
        x[self.base_test_case.treatment_variable.name] = [self.treatment_value, self.control_value]

        for k, v in self.adjustment_config.items():
            x[k] = v
        for k, v in self.effect_modifiers.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if str(x.dtypes[col]) == "object":
                x = pd.get_dummies(x, columns=[col], drop_first=True)

        return model.get_prediction(x).summary_frame()

    def to_dict(self) -> dict:
        """
        Convert the estimator to a python dictionary for easy serialisation as JSON or CSV.

        :returns: A JSON serialisable dict representing the estimator.
        """
        result = super().to_dict()
        if self.adjustment_config:
            result["adjustment_config"] = self.adjustment_config
        if self.formula:
            result["formula"] = self.formula
        return result
