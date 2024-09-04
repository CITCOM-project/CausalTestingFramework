"""This module contains the Estimator abstract class, as well as its concrete extensions: LogisticRegressionEstimator,
LinearRegressionEstimator"""

import logging
from abc import ABC, abstractmethod
from typing import Any
from math import ceil

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix  # pylint: disable = no-name-in-module
from patsy import ModelDesc
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from lifelines import CoxPHFitter

from causal_testing.specification.variable import Variable
from causal_testing.specification.capabilities import TreatmentSequence, Capability

logger = logging.getLogger(__name__)


class Estimator(ABC):
    # pylint: disable=too-many-instance-attributes
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
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        df: pd.DataFrame = None,
        effect_modifiers: dict[str:Any] = None,
        alpha: float = 0.05,
        query: str = "",
    ):
        self.treatment = treatment
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.adjustment_set = adjustment_set
        self.outcome = outcome
        self.alpha = alpha
        self.df = df.query(query) if query else df

        if effect_modifiers is None:
            self.effect_modifiers = {}
        elif isinstance(effect_modifiers, dict):
            self.effect_modifiers = effect_modifiers
        else:
            raise ValueError(f"Unsupported type for effect_modifiers {effect_modifiers}. Expected iterable")
        self.modelling_assumptions = []
        if query:
            self.modelling_assumptions.append(query)
        self.add_modelling_assumptions()
        logger.debug("Effect Modifiers: %s", self.effect_modifiers)

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """

    def compute_confidence_intervals(self) -> list[float, float]:
        """
        Estimate the 95% Wald confidence intervals for the effect of changing the treatment from control values to
        treatment values on the outcome.
        :return: 95% Wald confidence intervals.
        """


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


class LinearRegressionEstimator(Estimator):
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
            treatment,
            treatment_value,
            control_value,
            adjustment_set,
            outcome,
            df,
            effect_modifiers,
            alpha=alpha,
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

    def estimate_coefficient(self) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the unit average treatment effect of the treatment on the outcome. That is, the change in outcome
        caused by a unit change in treatment.

        :return: The unit average treatment effect and the 95% Wald confidence intervals.
        """
        model = self._run_linear_regression()
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
        model = self._run_linear_regression()

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

    def estimate_control_treatment(self, adjustment_config: dict = None) -> tuple[pd.Series, pd.Series]:
        """Estimate the outcomes under control and treatment.

        :return: The estimated outcome under control and treatment in the form
        (control_outcome, treatment_outcome).
        """
        if adjustment_config is None:
            adjustment_config = {}
        model = self._run_linear_regression()

        x = pd.DataFrame(columns=self.df.columns)
        x[self.treatment] = [self.treatment_value, self.control_value]
        x["Intercept"] = 1  # self.intercept
        for k, v in adjustment_config.items():
            x[k] = v
        for k, v in self.effect_modifiers.items():
            x[k] = v
        x = dmatrix(self.formula.split("~")[1], x, return_type="dataframe")
        for col in x:
            if str(x.dtypes[col]) == "object":
                x = pd.get_dummies(x, columns=[col], drop_first=True)
        x = x[model.params.index]
        y = model.get_prediction(x).summary_frame()

        return y.iloc[1], y.iloc[0]

    def estimate_risk_ratio(self, adjustment_config: dict = None) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the risk_ratio effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        if adjustment_config is None:
            adjustment_config = {}
        control_outcome, treatment_outcome = self.estimate_control_treatment(adjustment_config=adjustment_config)
        ci_low = pd.Series(treatment_outcome["mean_ci_lower"] / control_outcome["mean_ci_upper"])
        ci_high = pd.Series(treatment_outcome["mean_ci_upper"] / control_outcome["mean_ci_lower"])
        return pd.Series(treatment_outcome["mean"] / control_outcome["mean"]), [ci_low, ci_high]

    def estimate_ate_calculated(self, adjustment_config: dict = None) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """Estimate the ate effect of the treatment on the outcome. That is, the change in outcome caused
        by changing the treatment variable from the control value to the treatment value. Here, we actually
        calculate the expected outcomes under control and treatment and divide one by the other. This
        allows for custom terms to be put in such as squares, inverses, products, etc.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        if adjustment_config is None:
            adjustment_config = {}
        control_outcome, treatment_outcome = self.estimate_control_treatment(adjustment_config=adjustment_config)
        ci_low = pd.Series(treatment_outcome["mean_ci_lower"] - control_outcome["mean_ci_upper"])
        ci_high = pd.Series(treatment_outcome["mean_ci_upper"] - control_outcome["mean_ci_lower"])
        return pd.Series(treatment_outcome["mean"] - control_outcome["mean"]), [ci_low, ci_high]

    def _run_linear_regression(self) -> RegressionResultsWrapper:
        """Run linear regression of the treatment and adjustment set against the outcome and return the model.

        :return: The model after fitting to data.
        """
        model = smf.ols(formula=self.formula, data=self.df).fit()
        self.model = model
        return model

    def _get_confidence_intervals(self, model, treatment):
        confidence_intervals = model.conf_int(alpha=self.alpha, cols=None)
        ci_low, ci_high = (
            pd.Series(confidence_intervals[0].loc[treatment]),
            pd.Series(confidence_intervals[1].loc[treatment]),
        )
        return [ci_low, ci_high]


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


class InstrumentalVariableEstimator(Estimator):
    """
    Carry out estimation using instrumental variable adjustment rather than conventional adjustment. This means we do
    not need to observe all confounders in order to adjust for them. A key assumption here is linearity.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        instrument: str,
        df: pd.DataFrame = None,
        intercept: int = 1,
        effect_modifiers: dict = None,  # Not used (yet?). Needed for compatibility
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
            effect_modifiers=None,
            alpha=alpha,
            query=query,
        )
        self.intercept = intercept
        self.model = None
        self.instrument = instrument

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions.append(
            """The instrument and the treatment, and the treatment and the outcome must be
        related linearly in the form Y = aX + b."""
        )
        self.modelling_assumptions.append(
            """The three IV conditions must hold
            (i) Instrument is associated with treatment
            (ii) Instrument does not affect outcome except through its potential effect on treatment
            (iii) Instrument and outcome do not share causes
        """
        )

    def estimate_iv_coefficient(self, df) -> float:
        """
        Estimate the linear regression coefficient of the treatment on the
        outcome.
        """
        # Estimate the total effect of instrument I on outcome Y = abI + c1
        ab = sm.OLS(df[self.outcome], df[[self.instrument]]).fit().params[self.instrument]

        # Estimate the direct effect of instrument I on treatment X = aI + c1
        a = sm.OLS(df[self.treatment], df[[self.instrument]]).fit().params[self.instrument]

        # Estimate the coefficient of I on X by cancelling
        return ab / a

    def estimate_coefficient(self, bootstrap_size=100) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """
        Estimate the unit ate (i.e. coefficient) of the treatment on the
        outcome.
        """
        bootstraps = sorted(
            [self.estimate_iv_coefficient(self.df.sample(len(self.df), replace=True)) for _ in range(bootstrap_size)]
        )
        bound = ceil((bootstrap_size * self.alpha) / 2)
        ci_low = pd.Series(bootstraps[bound])
        ci_high = pd.Series(bootstraps[bootstrap_size - bound])

        return pd.Series(self.estimate_iv_coefficient(self.df)), [ci_low, ci_high]


class IPCWEstimator(Estimator):
    """
    Class to perform inverse probability of censoring weighting (IPCW) estimation
    for sequences of treatments over time-varying data.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        df: pd.DataFrame,
        timesteps_per_intervention: int,
        control_strategy: TreatmentSequence,
        treatment_strategy: TreatmentSequence,
        outcome: str,
        fault_column: str,
        fit_bl_switch_formula: str,
        fit_bltd_switch_formula: str,
        eligibility=None,
        alpha: float = 0.05,
    ):
        super().__init__(
            [c.variable for c in treatment_strategy.capabilities],
            [c.value for c in treatment_strategy.capabilities],
            [c.value for c in control_strategy.capabilities],
            None,
            outcome,
            df,
            None,
            alpha=alpha,
            query="",
        )
        self.timesteps_per_intervention = timesteps_per_intervention
        self.control_strategy = control_strategy
        self.treatment_strategy = treatment_strategy
        self.outcome = outcome
        self.fault_column = fault_column
        self.timesteps_per_intervention = timesteps_per_intervention
        self.fit_bl_switch_formula = fit_bl_switch_formula
        self.fit_bltd_switch_formula = fit_bltd_switch_formula
        self.eligibility = eligibility
        self.df = df
        self.preprocess_data()

    def add_modelling_assumptions(self):
        self.modelling_assumptions.append("The variables in the data vary over time.")

    def setup_xo_t_do(self, strategy_assigned: list, strategy_followed: list, eligible: pd.Series):
        """
        Return a binary sequence with each bit representing whether the current
        index is the time point at which the individual diverted from the
        assigned treatment strategy (and thus should be censored).

        :param strategy_assigned - the assigned treatment strategy
        :param strategy_followed - the strategy followed by the individual
        :param eligible - binary sequence represnting the eligibility of the individual at each time step
        """
        strategy_assigned = [1] + strategy_assigned + [1]
        strategy_followed = [1] + strategy_followed + [1]

        mask = (
            pd.Series(strategy_assigned, index=eligible.index) != pd.Series(strategy_followed, index=eligible.index)
        ).astype("boolean")
        mask = mask | ~eligible
        mask.reset_index(inplace=True, drop=True)
        false = mask.loc[mask]
        if false.empty:
            return np.zeros(len(mask))
        mask = (mask * 1).tolist()
        cutoff = false.index[0] + 1
        return mask[:cutoff] + ([None] * (len(mask) - cutoff))

    def setup_fault_t_do(self, individual: pd.DataFrame):
        """
        Return a binary sequence with each bit representing whether the current
        index is the time point at which the event of interest (i.e. a fault)
        occurred.
        """
        fault = individual[~individual[self.fault_column]]
        fault_t_do = pd.Series(np.zeros(len(individual)), index=individual.index)

        if not fault.empty:
            fault_time = individual["time"].loc[fault.index[0]]
            # Ceiling to nearest observation point
            fault_time = ceil(fault_time / self.timesteps_per_intervention) * self.timesteps_per_intervention
            # Set the correct observation point to be the fault time of doing (fault_t_do)
            observations = individual.loc[
                (individual["time"] % self.timesteps_per_intervention == 0) & (individual["time"] < fault_time)
            ]
            if not observations.empty:
                fault_t_do.loc[observations.index[0]] = 1
        assert sum(fault_t_do) <= 1, f"Multiple fault times for\n{individual}"

        return pd.DataFrame({"fault_t_do": fault_t_do})

    def setup_fault_time(self, individual: pd.DataFrame, perturbation: float = -0.001):
        """
        Return the time at which the event of interest (i.e. a fault) occurred.
        """
        fault = individual[~individual[self.fault_column]]
        fault_time = (
            individual["time"].loc[fault.index[0]]
            if not fault.empty
            else (individual["time"].max() + self.timesteps_per_intervention)
        )
        return pd.DataFrame({"fault_time": np.repeat(fault_time + perturbation, len(individual))})

    def preprocess_data(self):
        """
        Set up the treatment-specific columns in the data that are needed to estimate the hazard ratio.
        """
        self.df["trtrand"] = None  # treatment/control arm
        self.df["xo_t_do"] = None  # did the individual deviate from the treatment of interest here?
        self.df["eligible"] = self.df.eval(self.eligibility) if self.eligibility is not None else True

        # when did a fault occur?
        self.df["fault_time"] = self.df.groupby("id")[[self.fault_column, "time"]].apply(self.setup_fault_time).values
        self.df["fault_t_do"] = (
            self.df.groupby("id")[["id", "time", self.fault_column]].apply(self.setup_fault_t_do).values
        )
        assert not pd.isnull(self.df["fault_time"]).any()

        living_runs = self.df.query("fault_time > 0").loc[
            (self.df["time"] % self.timesteps_per_intervention == 0)
            & (self.df["time"] <= self.control_strategy.total_time())
        ]

        individuals = []
        new_id = 0
        logging.debug("  Preprocessing groups")
        for _, individual in living_runs.groupby("id"):
            assert sum(individual["fault_t_do"]) <= 1, (
                f"Error initialising fault_t_do for individual\n"
                f"{individual[['id', 'time', 'fault_time', 'fault_t_do']]}\n"
                "with fault at {individual.fault_time.iloc[0]}"
            )

            strategy_followed = [
                Capability(
                    c.variable,
                    individual.loc[individual["time"] == c.start_time, c.variable].values[0],
                    c.start_time,
                    c.end_time,
                )
                for c in self.treatment_strategy.capabilities
            ]

            # Control flow:
            # Individuals that start off in both arms, need cloning (hence incrementing the ID within the if statement)
            # Individuals that don't start off in either arm are left out
            for inx, strategy_assigned in [(0, self.control_strategy), (1, self.treatment_strategy)]:
                if strategy_assigned.capabilities[0] == strategy_followed[0] and individual.eligible.iloc[0]:
                    individual["id"] = new_id
                    new_id += 1
                    individual["trtrand"] = inx
                    individual["xo_t_do"] = self.setup_xo_t_do(
                        strategy_assigned.capabilities, strategy_followed, individual["eligible"]
                    )
                    individuals.append(individual.loc[individual["time"] <= individual["fault_time"]].copy())
        if len(individuals) == 0:
            raise ValueError("No individuals followed either strategy.")

        self.df = pd.concat(individuals)

    def estimate_hazard_ratio(self):
        """
        Estimate the hazard ratio.
        """

        if self.df["fault_t_do"].sum() == 0:
            raise ValueError("No recorded faults")

        preprocessed_data = self.df.loc[self.df["xo_t_do"] == 0].copy()

        # Use logistic regression to predict switching given baseline covariates
        fit_bl_switch = smf.logit(self.fit_bl_switch_formula, data=self.df).fit()

        preprocessed_data["pxo1"] = fit_bl_switch.predict(preprocessed_data)

        # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
        fit_bltd_switch = smf.logit(
            self.fit_bltd_switch_formula,
            data=self.df,
        ).fit()

        preprocessed_data["pxo2"] = fit_bltd_switch.predict(preprocessed_data)

        # IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
        # Estimate the probabilities of remaining ‘un-switched’ and hence the weights

        preprocessed_data["num"] = 1 - preprocessed_data["pxo1"]
        preprocessed_data["denom"] = 1 - preprocessed_data["pxo2"]
        preprocessed_data[["num", "denom"]] = (
            preprocessed_data.sort_values(["id", "time"]).groupby("id")[["num", "denom"]].cumprod()
        )

        assert (
            not preprocessed_data["num"].isnull().any()
        ), f"{len(preprocessed_data['num'].isnull())} null numerator values"
        assert (
            not preprocessed_data["denom"].isnull().any()
        ), f"{len(preprocessed_data['denom'].isnull())} null denom values"

        preprocessed_data["weight"] = 1 / preprocessed_data["denom"]
        preprocessed_data["sweight"] = preprocessed_data["num"] / preprocessed_data["denom"]

        preprocessed_data["tin"] = preprocessed_data["time"]
        preprocessed_data["tout"] = pd.concat(
            [(preprocessed_data["time"] + self.timesteps_per_intervention), preprocessed_data["fault_time"]],
            axis=1,
        ).min(axis=1)

        assert (preprocessed_data["tin"] <= preprocessed_data["tout"]).all(), (
            f"Left before joining\n" f"{preprocessed_data.loc[preprocessed_data['tin'] >= preprocessed_data['tout']]}"
        )

        #  IPCW step 4: Use these weights in a weighted analysis of the outcome model
        # Estimate the KM graph and IPCW hazard ratio using Cox regression.
        cox_ph = CoxPHFitter(alpha=self.alpha)
        cox_ph.fit(
            df=preprocessed_data,
            duration_col="tout",
            event_col="fault_t_do",
            weights_col="weight",
            cluster_col="id",
            robust=True,
            formula="trtrand",
            entry_col="tin",
        )

        ci_low, ci_high = [np.exp(cox_ph.confidence_intervals_)[col] for col in cox_ph.confidence_intervals_.columns]

        return (cox_ph.hazard_ratios_, (ci_low, ci_high))
