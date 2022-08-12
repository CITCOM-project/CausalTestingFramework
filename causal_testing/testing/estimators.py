import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.regression.linear_model import RegressionResultsWrapper

from causal_testing.specification.variable import Variable

logger = logging.getLogger(__name__)


class Estimator(ABC):
    """An estimator contains all of the information necessary to compute a causal estimate for the effect of changing
    a set of treatment variables to a set of values.

    All estimators must implement the following two methods:

    1) add_modelling_assumptions: The validity of a model-assisted causal inference result depends on whether
    the modelling assumptions imposed by a model actually hold. Therefore, for each model, is important to state
    the modelling assumption upon which the validity of the results depend. To achieve this, the estimator object
    maintains a list of modelling assumptions (as strings). If a user wishes to implement their own estimator, they
    must implement this method and add all assumptions to the list of modelling assumptions.

    2) estimate_ate: All estimators must be capable of returning the average treatment effect as a minimum. That is, the
    average effect of the intervention (changing treatment from control to treated value) on the outcome of interest
    adjusted for all confounders.
    """

    def __init__(
        self,
        treatment: tuple,
        treatment_values: float,
        control_values: float,
        adjustment_set: set,
        outcome: tuple,
        df: pd.DataFrame = None,
        effect_modifiers: dict[Variable:Any] = None,
    ):
        self.treatment = treatment
        self.treatment_values = treatment_values
        self.control_values = control_values
        self.adjustment_set = adjustment_set
        self.outcome = outcome
        self.df = df
        if effect_modifiers is None:
            self.effect_modifiers = dict()
        elif isinstance(effect_modifiers, set) or isinstance(effect_modifiers, list):
            self.effect_modifiers = {k.name for k in effect_modifiers}
        elif isinstance(effect_modifiers, dict):
            self.effect_modifiers = {k.name: v for k, v in effect_modifiers.items()}
        else:
            raise ValueError(f"Unsupported type for effect_modifiers {effect_modifiers}. Expected iterable")
        self.modelling_assumptions = []
        logger.debug("Effect Modifiers: %s", self.effect_modifiers)

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        pass

    @abstractmethod
    def estimate_ate(self) -> float:
        """
        Estimate the unit effect of the treatment on the outcome. That is, the coefficient of the treatment variable
        in the linear regression equation.
        :return: The intercept and coefficient of the linear regression equation
        """
        pass

    def compute_confidence_intervals(self) -> list[float, float]:
        """
        Estimate the 95% Wald confidence intervals for the effect of changing the treatment from control values to
        treatment values on the outcome.
        :return: 95% Wald confidence intervals.
        """
        pass


class LogisticRegressionEstimator(Estimator):
    """A Logistic Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear). It is designed
    for estimating categorical outcomes.
    """

    def __init__(
        self,
        treatment: tuple,
        treatment_values: float,
        control_values: float,
        adjustment_set: set,
        outcome: tuple,
        df: pd.DataFrame = None,
        effect_modifiers: dict[Variable:Any] = None,
        intercept: int = 1,
    ):
        super().__init__(treatment, treatment_values, control_values, adjustment_set, outcome, df, effect_modifiers)

        for term in self.effect_modifiers:
            self.adjustment_set.add(term)

        self.product_terms = []
        self.square_terms = []
        self.inverse_terms = []
        self.intercept = intercept

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += (
            "The variables in the data must fit a shape which can be expressed as a linear"
            "combination of parameters and functions of variables. Note that these functions"
            "do not need to be linear."
        )
        self.modelling_assumptions += "The outcome must be binary."
        self.modelling_assumptions += "Independently and identically distributed errors."

    def _run_logistic_regression(self) -> RegressionResultsWrapper:
        """Run logistic regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        # 1. Reduce dataframe to contain only the necessary columns
        reduced_df = self.df.copy()
        necessary_cols = list(self.treatment) + list(self.adjustment_set) + list(self.outcome)
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]
        reduced_df = reduced_df.sort_values(list(self.treatment))
        logger.debug(reduced_df[necessary_cols])

        # 2. Add intercept
        reduced_df["Intercept"] = self.intercept

        # 3. Estimate the unit difference in outcome caused by unit difference in treatment
        cols = list(self.treatment)
        cols += [x for x in self.adjustment_set if x not in cols]
        treatment_and_adjustments_cols = reduced_df[cols + ["Intercept"]]
        outcome_col = reduced_df[list(self.outcome)]
        regression = sm.Logit(outcome_col, treatment_and_adjustments_cols)
        model = regression.fit()
        return model

    def estimate_control_treatment(self) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_logistic_regression()
        self.model = model

        x = pd.DataFrame()
        x[self.treatment[0]] = [self.treatment_values, self.control_values]
        x["Intercept"] = self.intercept
        for k, v in self.effect_modifiers.items():
            x[k] = v
        for t in self.square_terms:
            x[t + "^2"] = x[t] ** 2
        for t in self.inverse_terms:
            x["1/" + t] = 1 / x[t]
        for a, b in self.product_terms:
            x[f"{a}*{b}"] = x[a] * x[b]
        x = x[model.params.index]

        y = model.predict(x)
        return y.iloc[1], y.iloc[0]

    def estimate_ate(self) -> float:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and take one away from the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The average treatment effect. Confidence intervals are not yet supported.
        """
        control_outcome, treatment_outcome = self.estimate_control_treatment()

        return treatment_outcome - control_outcome

    def estimate_risk_ratio(self) -> float:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The average treatment effect. Confidence intervals are not yet supported.
        """
        control_outcome, treatment_outcome = self.estimate_control_treatment()

        return treatment_outcome / control_outcome

    def estimate_unit_odds_ratio(self) -> float:
        """Estimate the odds ratio of increasing the treatment by one. In logistic regression, this corresponds to the
        coefficient of the treatment of interest.

        :return: The odds ratio. Confidence intervals are not yet supported.
        """
        model = self._run_logistic_regression()
        return np.exp(model.params[self.treatment[0]])


class LinearRegressionEstimator(Estimator):
    """A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    def __init__(
        self,
        treatment: tuple,
        treatment_values: float,
        control_values: float,
        adjustment_set: set,
        outcome: tuple,
        df: pd.DataFrame = None,
        effect_modifiers: dict[Variable:Any] = None,
        product_terms: list[tuple[Variable, Variable]] = None,
        intercept: int = 1,
    ):
        super().__init__(treatment, treatment_values, control_values, adjustment_set, outcome, df, effect_modifiers)

        if product_terms is None:
            product_terms = []
        for (term_a, term_b) in product_terms:
            self.add_product_term_to_df(term_a, term_b)
        for term in self.effect_modifiers:
            self.adjustment_set.add(term)

        self.product_terms = product_terms
        self.square_terms = []
        self.inverse_terms = []
        self.intercept = intercept

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += (
            "The variables in the data must fit a shape which can be expressed as a linear"
            "combination of parameters and functions of variables. Note that these functions"
            "do not need to be linear."
        )

    def add_squared_term_to_df(self, term_to_square: str):
        """Add a squared term to the linear regression model and df.

        This enables the user to capture curvilinear relationships with a linear regression model, not just straight
        lines, while automatically adding the modelling assumption imposed by the addition of this term.

        :param term_to_square: The term (column in data and variable in DAG) which is to be squared.
        """
        new_term = str(term_to_square) + "^2"
        self.df[new_term] = self.df[term_to_square] ** 2
        self.adjustment_set.add(new_term)
        self.modelling_assumptions += (
            f"Relationship between {self.treatment} and {self.outcome} varies quadratically" f"with {term_to_square}."
        )
        self.square_terms.append(term_to_square)

    def add_inverse_term_to_df(self, term_to_invert: str):
        """Add an inverse term to the linear regression model and df.

        This enables the user to capture curvilinear relationships with a linear regression model, not just straight
        lines, while automatically adding the modelling assumption imposed by the addition of this term.

        :param term_to_square: The term (column in data and variable in DAG) which is to be squared.
        """
        new_term = "1/" + str(term_to_invert)
        self.df[new_term] = 1 / self.df[term_to_invert]
        self.adjustment_set.add(new_term)
        self.modelling_assumptions += (
            f"Relationship between {self.treatment} and {self.outcome} varies inversely" f"with {term_to_invert}."
        )
        self.inverse_terms.append(term_to_invert)

    def add_product_term_to_df(self, term_a: str, term_b: str):
        """Add a product term to the linear regression model and df.

        This enables the user to capture interaction between a pair of variables in the model. In other words, while
        each covariate's contribution to the mean is assumed to be independent of the other covariates, the pair of
        product terms term_a*term_b a are restricted to vary linearly with each other.

        :param term_a: The first term of the product term.
        :param term_b: The second term of the product term.
        """
        new_term = str(term_a) + "*" + str(term_b)
        self.df[new_term] = self.df[term_a] * self.df[term_b]
        self.adjustment_set.add(new_term)
        self.modelling_assumptions += f"{term_a} and {term_b} vary linearly with each other."
        self.product_terms.append((term_a, term_b))

    def estimate_unit_ate(self) -> float:
        """Estimate the unit average treatment effect of the treatment on the outcome. That is, the change in outcome
        caused by a unit change in treatment.

        :return: The unit average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
        unit_effect = model.params[list(self.treatment)].values[0]  # Unit effect is the coefficient of the treatment
        [ci_low, ci_high] = self._get_confidence_intervals(model)
        return unit_effect * self.treatment_values - unit_effect * self.control_values, [ci_low, ci_high]

    def estimate_ate(self) -> tuple[float, list[float, float], float]:
        """Estimate the average treatment effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
        # Create an empty individual for the control and treated
        individuals = pd.DataFrame(1, index=["control", "treated"], columns=model.params.index)
        individuals.loc["control", list(self.treatment)] = self.control_values
        individuals.loc["treated", list(self.treatment)] = self.treatment_values
        # This is a temporary hack
        for t in self.square_terms:
            individuals[t + "^2"] = individuals[t] ** 2
        for a, b in self.product_terms:
            individuals[f"{a}*{b}"] = individuals[a] * individuals[b]

        # Perform a t-test to compare the predicted outcome of the control and treated individual (ATE)
        t_test_results = model.t_test(individuals.loc["treated"] - individuals.loc["control"])
        ate = t_test_results.effect[0]
        confidence_intervals = list(t_test_results.conf_int().flatten())
        return ate, confidence_intervals

    def estimate_control_treatment(self) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
        self.model = model

        x = pd.DataFrame()
        x[self.treatment[0]] = [self.treatment_values, self.control_values]
        x["Intercept"] = self.intercept
        for k, v in self.effect_modifiers.items():
            x[k] = v
        for t in self.square_terms:
            x[t + "^2"] = x[t] ** 2
        for t in self.inverse_terms:
            x["1/" + t] = 1 / x[t]
        for a, b in self.product_terms:
            x[f"{a}*{b}"] = x[a] * x[b]
        x = x[model.params.index]

        y = model.get_prediction(x).summary_frame()
        return y.iloc[1], y.iloc[0]

    def estimate_risk_ratio(self) -> tuple[float, list[float, float]]:
        """Estimate the risk_ratio effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        control_outcome, treatment_outcome = self.estimate_control_treatment()
        ci_low = treatment_outcome["mean_ci_lower"] / control_outcome["mean_ci_upper"]
        ci_high = treatment_outcome["mean_ci_upper"] / control_outcome["mean_ci_lower"]

        return (treatment_outcome["mean"] / control_outcome["mean"]), [ci_low, ci_high]

    def estimate_ate_calculated(self) -> tuple[float, list[float, float]]:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        control_outcome, treatment_outcome = self.estimate_control_treatment()
        ci_low = treatment_outcome["mean_ci_lower"] - control_outcome["mean_ci_upper"]
        ci_high = treatment_outcome["mean_ci_upper"] - control_outcome["mean_ci_lower"]

        return (treatment_outcome["mean"] - control_outcome["mean"]), [ci_low, ci_high]

    def estimate_cates(self) -> tuple[float, list[float, float]]:
        """Estimate the conditional average treatment effect of the treatment on the outcome. That is, the change
        in outcome caused by changing the treatment variable from the control value to the treatment value.

        :return: The conditional average treatment effect and the 95% Wald confidence intervals.
        """
        assert (
            self.effect_modifiers
        ), f"Must have at least one effect modifier to compute CATE - {self.effect_modifiers}."
        x = pd.DataFrame()
        x[self.treatment[0]] = [self.treatment_values, self.control_values]
        x["Intercept"] = self.intercept
        for k, v in self.effect_modifiers.items():
            self.adjustment_set.add(k)
            x[k] = v
        if hasattr(self, "square_terms"):
            for t in self.square_terms:
                x[t + "^2"] = x[t] ** 2
        if hasattr(self, "product_terms"):
            for a, b in self.product_terms:
                x[f"{a}*{b}"] = x[a] * x[b]

        model = self._run_linear_regression()
        y = model.predict(x)
        treatment_outcome = y.iloc[0]
        control_outcome = y.iloc[1]

        return treatment_outcome - control_outcome, None

    def _run_linear_regression(self) -> RegressionResultsWrapper:
        """Run linear regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        # 1. Reduce dataframe to contain only the necessary columns
        reduced_df = self.df.copy()
        necessary_cols = list(self.treatment) + list(self.adjustment_set) + list(self.outcome)
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]
        reduced_df = reduced_df.sort_values(list(self.treatment))
        logger.debug(reduced_df[necessary_cols])

        # 2. Add intercept
        reduced_df["Intercept"] = self.intercept

        # 3. Estimate the unit difference in outcome caused by unit difference in treatment
        cols = list(self.treatment)
        cols += [x for x in self.adjustment_set if x not in cols]
        treatment_and_adjustments_cols = reduced_df[cols + ["Intercept"]]
        outcome_col = reduced_df[list(self.outcome)]
        regression = sm.OLS(outcome_col, treatment_and_adjustments_cols)
        model = regression.fit()
        return model

    def _get_confidence_intervals(self, model):
        confidence_intervals = model.conf_int(alpha=0.05, cols=None)
        ci_low, ci_high = confidence_intervals[0][list(self.treatment)], confidence_intervals[1][list(self.treatment)]
        return [ci_low.values[0], ci_high.values[0]]


class CausalForestEstimator(Estimator):
    """A causal random forest estimator is a non-parametric estimator which recursively partitions the covariate space
    to learn a low-dimensional representation of treatment effect heterogeneity. This form of estimator is best suited
    to the estimation of heterogeneous treatment effects i.e. the estimated effect for every sample rather than the
    population average.
    """

    def add_modelling_assumptions(self):
        """Add any modelling assumptions to the estimator.

        :return self: Update self.modelling_assumptions
        """
        self.modelling_assumptions += "Non-parametric estimator: no restrictions imposed on the data."

    def estimate_ate(self) -> float:
        """Estimate the average treatment effect.

        :return ate, confidence_intervals: The average treatment effect and 95% confidence intervals.
        """
        # Remove any NA containing rows
        reduced_df = self.df.copy()
        necessary_cols = list(self.treatment) + list(self.adjustment_set) + list(self.outcome)
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]

        # Split data into effect modifiers (X), confounders (W), treatments (T), and outcome (Y)
        # TODO: Is it right to ignore the adjustment set if we have effect modifiers?
        if self.effect_modifiers:
            effect_modifier_df = reduced_df[list(self.effect_modifiers)]
        else:
            effect_modifier_df = reduced_df[list(self.adjustment_set)]
        confounders_df = reduced_df[list(self.adjustment_set)]
        treatment_df = np.ravel(reduced_df[list(self.treatment)])
        outcome_df = np.ravel(reduced_df[list(self.outcome)])

        # Fit the model to the data using a gradient boosting regressor for both the treatment and outcome model
        model = CausalForestDML(
            model_y=GradientBoostingRegressor(),
            model_t=GradientBoostingRegressor(),
        )
        model.fit(outcome_df, treatment_df, X=effect_modifier_df, W=confounders_df)

        # Obtain the ATE and 95% confidence intervals
        ate = model.ate(effect_modifier_df, T0=self.control_values, T1=self.treatment_values)
        ate_interval = model.ate_interval(effect_modifier_df, T0=self.control_values, T1=self.treatment_values)
        ci_low, ci_high = ate_interval[0], ate_interval[1]
        return ate, [ci_low, ci_high]

    def estimate_cates(self) -> pd.DataFrame:
        """Estimate the conditional average treatment effect for each sample in the data as a function of a set of
        covariates (X) i.e. effect modifiers. That is, the predicted change in outcome caused by the intervention
        (change in treatment from control to treatment value) for every execution of the system-under-test, taking into
        account the value of each effect modifier X. As a result, for every unique setting of the set of covariates X,
        we expect a different CATE.

        :return results_df: A dataframe containing a conditional average treatment effect, 95% confidence intervals, and
        the covariate (effect modifier) values for each sample.
        """

        # Remove any NA containing rows
        reduced_df = self.df.copy()
        necessary_cols = list(self.treatment) + list(self.adjustment_set) + list(self.outcome)
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]

        # Split data into effect modifiers (X), confounders (W), treatments (T), and outcome (Y)
        if self.effect_modifiers:
            effect_modifier_df = reduced_df[list(self.effect_modifiers)]
        else:
            raise Exception("CATE requires the user to define a set of effect modifiers.")

        if self.adjustment_set:
            confounders_df = reduced_df[list(self.adjustment_set)]
        else:
            confounders_df = None
        treatment_df = reduced_df[list(self.treatment)]
        outcome_df = reduced_df[list(self.outcome)]

        # Fit a model to the data
        model = CausalForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor())
        model.fit(outcome_df, treatment_df, X=effect_modifier_df, W=confounders_df)

        # Obtain CATES and confidence intervals
        conditional_ates = model.effect(effect_modifier_df, T0=self.control_values, T1=self.treatment_values).flatten()
        [ci_low, ci_high] = model.effect_interval(
            effect_modifier_df, T0=self.control_values, T1=self.treatment_values, alpha=0.05
        )

        # Merge results into a dataframe (CATE, confidence intervals, and effect modifier values)
        results_df = pd.DataFrame(columns=["cate", "ci_low", "ci_high"])
        results_df["cate"] = list(conditional_ates)
        results_df["ci_low"] = list(ci_low.flatten())
        results_df["ci_high"] = list(ci_high.flatten())
        effect_modifier_df.reset_index(drop=True, inplace=True)
        results_df[list(self.effect_modifiers)] = effect_modifier_df
        results_df.sort_values(by=list(self.effect_modifiers), inplace=True)
        return results_df, None
