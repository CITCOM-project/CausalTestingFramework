"""This module contains the RegressionEstimator, which is an abstract class for concrete regression estimators."""

import ast
import logging
from abc import abstractmethod
from typing import Any

import pandas as pd
from patsy import ModelDesc, dmatrices, dmatrix  # pylint: disable = no-name-in-module
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_testing.estimation.abstract_estimator import Estimator

logger = logging.getLogger(__name__)


class RegressionEstimator(Estimator):
    """A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment_variable: str,
        outcome_variable: str,
        control_value: float = None,
        treatment_value: float = None,
        adjustment_set: set[str] = None,
        adjustment_config: dict[str, Any] = None,
        formula: str = None,
        alpha: float = 0.05,
    ):
        # pylint: disable=R0801
        super().__init__(
            treatment_variable=treatment_variable,
            outcome_variable=outcome_variable,
            control_value=control_value,
            treatment_value=treatment_value,
            adjustment_set=adjustment_set,
            alpha=alpha,
        )

        if formula is not None:
            self.formula = formula
            self._adjustment_set_from_formula()
            if adjustment_set is not None and set(adjustment_set) != set(self.adjustment_set):
                raise ValueError(
                    f"Specified formula {self.formula} does not match specified adjustment set {adjustment_set}"
                )
        elif adjustment_set is not None:
            terms = [treatment_variable] + sorted(list(adjustment_set))
            self.formula = f"{outcome_variable} ~ {' + '.join(terms)}"
        else:
            raise ValueError("Please specify either a formula or an adjustment set.")

        self.adjustment_config = adjustment_config if adjustment_config is not None else {}
        if not set(self.adjustment_config).issubset(self.adjustment_set):
            raise ValueError(
                "Specified configuration for variables "
                f"{sorted([v for v in adjustment_config if v not in self.adjustment_set])} "
                f"which are not in the adjustment set {self.adjustment_set}."
            )

    def _get_adjusted_variables(self, tree: ast.AST) -> set[str]:
        """
        Recursively return variables in an AST.
        :returns: Set of all variables not used as part of a function.
        """
        if isinstance(tree, ast.Name) and tree.id != self.treatment_variable:
            return {tree.id}
        if isinstance(tree, ast.Expression):
            return self._get_adjusted_variables(tree.body)
        if isinstance(tree, ast.Call):
            return set().union(*[self._get_adjusted_variables(arg) for arg in tree.args])
        if isinstance(tree, ast.BinOp):
            return self._get_adjusted_variables(tree.left).union(self._get_adjusted_variables(tree.right))
        return set()

    def _adjustment_set_from_formula(self):
        """
        Set up the adjustment set from the formula string.
        """
        desc = ModelDesc.from_formula(self.formula)

        # Check that the outcome variable is the dependent variable specified in the formula
        if desc.lhs_termlist:
            if [self.outcome_variable] != [term.name() for term in desc.lhs_termlist]:
                raise ValueError(
                    f"Left hand side of formula {self.formula} does not match the specified outcome_variable "
                    f"{self.outcome_variable}."
                )
        # If no dependent variable is specified, make it the outcome variable
        else:
            self.formula = f"{self.outcome_variable} ~ {self.formula}"

        raw_factors = {factor.code for term in desc.rhs_termlist for factor in term.factors}

        adjustment_set = set()

        for code in raw_factors:
            tree = ast.parse(code, mode="eval")
            adjustment_set = adjustment_set.union(self._get_adjusted_variables(tree))

        self.adjustment_set = sorted(list(adjustment_set))

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
        model = self.regressor(df[self.outcome_variable], df[covariates]).fit(disp=0)
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
            if param == self.treatment_variable or param.startswith(self.treatment_variable + "[")
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
        x[self.treatment_variable] = [self.treatment_value, self.control_value]

        for k, v in self.adjustment_config.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if isinstance(x[col], pd.CategoricalDtype) or pd.api.types.is_object_dtype(x[col]):
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
