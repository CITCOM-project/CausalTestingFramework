"""This module contains the LinearRegressionEstimator for estimating continuous outcomes."""

import logging
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix, ModelDesc  # pylint: disable = no-name-in-module

from causal_testing.specification.variable import Variable
from causal_testing.estimation.genetic_programming_regression_fitter import GP
from causal_testing.estimation.abstract_regression_estimator import RegressionEstimator

logger = logging.getLogger(__name__)


class LinearRegressionEstimator(RegressionEstimator):
    """A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    regressor = smf.ols

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
        # pylint: disable=too-many-arguments
        super().__init__(
            treatment,
            treatment_value,
            control_value,
            adjustment_set,
            outcome,
            df,
            effect_modifiers,
            formula,
            alpha,
            query,
        )
        for term in self.effect_modifiers:
            self.adjustment_set.add(term)

    def gp_formula(
        self,
        ngen: int = 100,
        pop_size: int = 20,
        num_offspring: int = 10,
        max_order: int = 0,
        extra_operators: list = None,
        sympy_conversions: dict = None,
        seeds: list = None,
        seed: int = 0,
    ):
        # pylint: disable=too-many-arguments
        """
        Use Genetic Programming (GP) to infer the regression equation from the data.

        :param ngen: The maximum number of GP generations to run for.
        :param pop_size: The GP population size.
        :param num_offspring: The number of offspring per generation.
        :param max_order: The maximum polynomial order to use, e.g. `max_order=2` will give
                          polynomials of the form `ax^2 + bx + c`.
        :param extra_operators: Additional operators for the GP (defaults are +, *, log(x), and 1/x).
                                Operations should be of the form (fun, numArgs), e.g. (add, 2).
        :param sympy_conversions: Dictionary of conversions of extra_operators for sympy,
                                  e.g. `"mul": lambda *args_: "Mul({},{})".format(*args_)`.
        :param seeds: Seed individuals for the population (e.g. if you think that the relationship between X and Y is
                      probably logarithmic, you can put that in).
        :param seed: Random seed for the GP.
        """
        gp = GP(
            df=self.df,
            features=sorted(list(self.adjustment_set.union([self.treatment]))),
            outcome=self.outcome,
            extra_operators=extra_operators,
            sympy_conversions=sympy_conversions,
            seed=seed,
            max_order=max_order,
        )
        formula = gp.run_gp(ngen=ngen, pop_size=pop_size, num_offspring=num_offspring, seeds=seeds)
        formula = gp.simplify(formula)
        self.formula = f"{self.outcome} ~ I({formula}) - 1"

    def estimate_coefficient(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the unit average treatment effect of the treatment on the outcome. That is, the change in outcome
        caused by a unit change in treatment.

        :return: The unit average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_regression()
        newline = "\n"
        patsy_md = ModelDesc.from_formula(self.treatment)

        if any(
            (
                self.df.dtypes[factor.name()] == "object"
                for factor in patsy_md.rhs_termlist[1].factors
                # We want to remove this long term as it prevents us from discovering categoricals within I(...) blocks
                if factor.name() in self.df.dtypes
            )
        ):
            design_info = dmatrix(self.formula.split("~")[1], self.df).design_info
            treatment = design_info.column_names[design_info.term_name_slices[self.treatment]]
        else:
            treatment = [self.treatment]
        assert set(treatment).issubset(
            model.params.index.tolist()
        ), f"{treatment} not in\n{'  ' + str(model.params.index).replace(newline, newline + '  ')}"
        unit_effect = model.params[treatment]  # Unit effect is the coefficient of the treatment
        [ci_low, ci_high] = self._get_confidence_intervals(model, treatment)
        return unit_effect, [ci_low, ci_high]

    def estimate_ate(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the average treatment effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_regression()

        # Create an empty individual for the control and treated
        individuals = pd.DataFrame(1, index=["control", "treated"], columns=model.params.index)

        # For Pandas version > 2, we need to explicitly state that the dataframe takes floating-point values
        individuals = individuals.astype(float)

        # It is ABSOLUTELY CRITICAL that these go last, otherwise we can't index
        # the effect with "ate = t_test_results.effect[0]"
        individuals.loc["control", [self.treatment]] = self.control_value
        individuals.loc["treated", [self.treatment]] = self.treatment_value

        # Perform a t-test to compare the predicted outcome of the control and treated individual (ATE)
        t_test_results = model.t_test(individuals.loc["treated"] - individuals.loc["control"])
        ate = pd.Series(t_test_results.effect[0])
        confidence_intervals = list(t_test_results.conf_int(alpha=self.alpha).flatten())
        confidence_intervals = [pd.Series(interval) for interval in confidence_intervals]
        return ate, confidence_intervals

    def estimate_risk_ratio(self, adjustment_config: dict = None) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the risk_ratio effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        prediction = self._predict(adjustment_config=adjustment_config)
        control_outcome, treatment_outcome = prediction.iloc[1], prediction.iloc[0]
        ci_low = pd.Series(treatment_outcome["mean_ci_lower"] / control_outcome["mean_ci_upper"])
        ci_high = pd.Series(treatment_outcome["mean_ci_upper"] / control_outcome["mean_ci_lower"])
        return pd.Series(treatment_outcome["mean"] / control_outcome["mean"]), [ci_low, ci_high]

    def estimate_ate_calculated(self, adjustment_config: dict = None) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :param: adjustment_config: The configuration of the adjustment set as a dict mapping variable names to
                                   their values. N.B. Every variable in the adjustment set MUST have a value in
                                   order to estimate the outcome under control and treatment.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        prediction = self._predict(adjustment_config=adjustment_config)
        control_outcome, treatment_outcome = prediction.iloc[1], prediction.iloc[0]
        ci_low = pd.Series(treatment_outcome["mean_ci_lower"] - control_outcome["mean_ci_upper"])
        ci_high = pd.Series(treatment_outcome["mean_ci_upper"] - control_outcome["mean_ci_lower"])
        return pd.Series(treatment_outcome["mean"] - control_outcome["mean"]), [ci_low, ci_high]

    def _get_confidence_intervals(self, model, treatment):
        confidence_intervals = model.conf_int(alpha=self.alpha, cols=None)
        ci_low, ci_high = (
            pd.Series(confidence_intervals[0].loc[treatment]),
            pd.Series(confidence_intervals[1].loc[treatment]),
        )
        return [ci_low, ci_high]
