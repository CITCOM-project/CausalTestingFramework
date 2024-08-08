"""This module contains the LogisticRegressionEstimator class for estimating categorical outcomes."""

import logging
from typing import Any
from math import ceil

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix  # pylint: disable = no-name-in-module
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from causal_testing.estimation.estimator import Estimator

logger = logging.getLogger(__name__)


class LogisticRegressionEstimator(Estimator):
    """A Logistic Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear). It is designed
    for estimating categorical outcomes.
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
        effect_modifiers: dict[str:Any] = None,
        formula: str = None,
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
            terms = [treatment] + sorted(list(adjustment_set)) + sorted(list(self.effect_modifiers))
            self.formula = f"{outcome} ~ {'+'.join(((terms)))}"

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

    def _run_logistic_regression(self, data) -> RegressionResultsWrapper:
        """Run logistic regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        model = smf.logit(formula=self.formula, data=data).fit(disp=0)
        self.model = model
        return model

    def estimate(self, data: pd.DataFrame, adjustment_config: dict = None) -> RegressionResultsWrapper:
        """add terms to the dataframe and estimate the outcome from the data
        :param data: A pandas dataframe containing execution data from the system-under-test.
        :param adjustment_config: Dictionary containing the adjustment configuration of the adjustment set
        """
        if adjustment_config is None:
            adjustment_config = {}
        if set(self.adjustment_set) != set(adjustment_config):
            raise ValueError(
                f"Invalid adjustment configuration {adjustment_config}. Must specify values for {self.adjustment_set}"
            )

        model = self._run_logistic_regression(data)

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
        # x = x[model.params.index]
        return model.predict(x)

    def estimate_control_treatment(
        self, adjustment_config: dict = None, bootstrap_size: int = 100
    ) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The estimated control and treatment values and their confidence
        intervals in the form ((ci_low, control, ci_high), (ci_low, treatment, ci_high)).
        """
        if adjustment_config is None:
            adjustment_config = {}
        y = self.estimate(self.df, adjustment_config=adjustment_config)

        try:
            bootstrap_samples = [
                self.estimate(self.df.sample(len(self.df), replace=True), adjustment_config=adjustment_config)
                for _ in range(bootstrap_size)
            ]
            control, treatment = zip(*[(x.iloc[1], x.iloc[0]) for x in bootstrap_samples])
        except PerfectSeparationError:
            logger.warning(
                "Perfect separation detected, results not available. Cannot calculate confidence intervals for such "
                "a small dataset."
            )
            return (y.iloc[1], None), (y.iloc[0], None)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix detected. Confidence intervals not available. Try with a larger data set")
            return (y.iloc[1], None), (y.iloc[0], None)

        # Delta method confidence intervals from
        # https://stackoverflow.com/questions/47414842/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode
        # cov = model.cov_params()
        # gradient = (y * (1 - y) * x.T).T  # matrix of gradients for each observation
        # std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient.to_numpy()])
        # c = 1.96  # multiplier for confidence interval
        # upper = np.maximum(0, np.minimum(1, y + std_errors * c))
        # lower = np.maximum(0, np.minimum(1, y - std_errors * c))

        return (y.iloc[1], np.array(control)), (y.iloc[0], np.array(treatment))

    def estimate_ate(self, adjustment_config: dict = None, bootstrap_size: int = 100) -> float:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and take one away from the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The estimated average treatment effect and 95% confidence intervals
        """
        if adjustment_config is None:
            adjustment_config = {}
        (control_outcome, control_bootstraps), (
            treatment_outcome,
            treatment_bootstraps,
        ) = self.estimate_control_treatment(bootstrap_size=bootstrap_size, adjustment_config=adjustment_config)
        estimate = treatment_outcome - control_outcome

        if control_bootstraps is None or treatment_bootstraps is None:
            return estimate, (None, None)

        bootstraps = sorted(list(treatment_bootstraps - control_bootstraps))
        bound = int((bootstrap_size * self.alpha) / 2)
        ci_low = bootstraps[bound]
        ci_high = bootstraps[bootstrap_size - bound]

        logger.info(
            f"Changing {self.treatment} from {self.control_value} to {self.treatment_value} gives an estimated "
            f"ATE of {ci_low} < {estimate} < {ci_high}"
        )
        assert ci_low < estimate < ci_high, f"Expecting {ci_low} < {estimate} < {ci_high}"

        return estimate, (ci_low, ci_high)

    def estimate_risk_ratio(self, adjustment_config: dict = None, bootstrap_size: int = 100) -> float:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The estimated risk ratio and 95% confidence intervals.
        """
        if adjustment_config is None:
            adjustment_config = {}
        (control_outcome, control_bootstraps), (
            treatment_outcome,
            treatment_bootstraps,
        ) = self.estimate_control_treatment(bootstrap_size=bootstrap_size, adjustment_config=adjustment_config)
        estimate = treatment_outcome / control_outcome

        if control_bootstraps is None or treatment_bootstraps is None:
            return estimate, (None, None)

        bootstraps = sorted(list(treatment_bootstraps / control_bootstraps))
        bound = ceil((bootstrap_size * self.alpha) / 2)
        ci_low = bootstraps[bound]
        ci_high = bootstraps[bootstrap_size - bound]

        logger.info(
            f"Changing {self.treatment} from {self.control_value} to {self.treatment_value} gives an estimated "
            f"risk ratio of {ci_low} < {estimate} < {ci_high}"
        )
        assert ci_low < estimate < ci_high, f"Expecting {ci_low} < {estimate} < {ci_high}"

        return estimate, (ci_low, ci_high)

    def estimate_unit_odds_ratio(self) -> float:
        """Estimate the odds ratio of increasing the treatment by one. In logistic regression, this corresponds to the
        coefficient of the treatment of interest.

        :return: The odds ratio. Confidence intervals are not yet supported.
        """
        model = self._run_logistic_regression(self.df)
        return np.exp(model.params[self.treatment])
