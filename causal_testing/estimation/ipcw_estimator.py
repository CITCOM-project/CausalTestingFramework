"""This module contains the IPCWEstimator class, for estimating the time to a particular event"""

import logging
from typing import Any
from uuid import uuid4


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Variable

logger = logging.getLogger(__name__)


class IPCWEstimator(Estimator):
    """
    Class to perform Inverse Probability of Censoring Weighting (IPCW) estimation
    for sequences of treatments over time-varying data.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        df: pd.DataFrame,
        timesteps_per_observation: int,
        control_strategy: list[tuple[int, str, Any]],
        treatment_strategy: list[tuple[int, str, Any]],
        outcome: Variable,
        status_column: str,
        fit_bl_switch_formula: str,
        fit_bltd_switch_formula: str,
        eligibility=None,
        alpha: float = 0.05,
        total_time: float = None,
    ):
        """
        Initialise IPCWEstimator.

        :param: df: Input DataFrame containing time-varying data.
        :param: timesteps_per_observation: Number of timesteps per observation.
        :param: control_strategy: The control strategy, with entries of the form (timestep, variable, value).
        :param: treatment_strategy: The treatment strategy, with entries of the form (timestep, variable, value).
        :param: outcome: Name of the outcome column in the DataFrame.
        :param: status_column: Name of the status column in the DataFrame, which should be True for operating normally,
                               False for a fault.
        :param: fit_bl_switch_formula: Formula for fitting the baseline switch model.
        :param: fit_bltd_switch_formula: Formula for fitting the baseline time-dependent switch model.
        :param: eligibility: Function to determine eligibility for treatment. Defaults to None for "always eligible".
        :param: alpha: Significance level for hypothesis testing. Defaults to 0.05.
        :param: total_time: Total time for the analysis. Defaults to one plus the length of of the strategy (control or
                            treatment) with the most elements multiplied by `timesteps_per_observation`.
        """
        super().__init__(
            base_test_case=BaseTestCase(None, outcome),
            treatment_value=[val for _, _, val in treatment_strategy],
            control_value=[val for _, _, val in control_strategy],
            adjustment_set=None,
            df=df,
            effect_modifiers=None,
            alpha=alpha,
            query="",
        )
        self.timesteps_per_observation = timesteps_per_observation
        self.control_strategy = control_strategy
        self.treatment_strategy = treatment_strategy
        self.status_column = status_column
        self.fit_bl_switch_formula = fit_bl_switch_formula
        self.fit_bltd_switch_formula = fit_bltd_switch_formula
        self.eligibility = eligibility
        self.df = df.sort_values(["id", "time"])
        self.len_control_group = None
        self.len_treatment_group = None

        if total_time is None:
            total_time = (
                max(len(self.control_strategy), len(self.treatment_strategy)) + 1
            ) * self.timesteps_per_observation
        self.total_time = total_time
        self.preprocess_data()

    def add_modelling_assumptions(self):
        self.modelling_assumptions.append("The variables in the data vary over time.")

    def setup_xo_t_do(self, individual: pd.DataFrame, strategy_assigned: list):
        """
        Return a binary sequence with each bit representing whether the current
        index is the time point at which the individual diverted from the
        assigned treatment strategy (and thus should be censored).

        :param individual: DataFrame representing the individual.
        :param strategy_assigned: The assigned treatment strategy.
        """

        default = {t: (-1, -1) for t in individual["time"].values}

        strategy_assigned = default | {t: (var, val) for t, var, val in strategy_assigned}
        strategy_followed = default | {
            t: (var, individual.loc[individual["time"] == t, var].values[0])
            for t, var, val in self.treatment_strategy
            if t in individual["time"].values
        }

        strategy_assigned = sorted(
            [(t, var, val) for t, (var, val) in strategy_assigned.items() if t in individual["time"].values]
        )
        strategy_followed = sorted(
            [(t, var, val) for t, (var, val) in strategy_followed.items() if t in individual["time"].values]
        )

        mask = (
            pd.Series(strategy_assigned, index=individual.index) != pd.Series(strategy_followed, index=individual.index)
        ).astype("boolean")
        mask = mask | ~individual["eligible"]
        mask.reset_index(inplace=True, drop=True)
        false = mask.loc[mask]
        if false.empty:
            return pd.DataFrame(
                {
                    "id": [str(uuid4())] * len(individual),
                    "xo_t_do": np.zeros(len(mask)),
                }
            )
        mask = (mask * 1).tolist()
        cutoff = false.index[0] + 1

        return pd.DataFrame(
            {
                "id": [str(uuid4())] * len(individual),
                "xo_t_do": pd.Series(mask[:cutoff] + ([None] * (len(mask) - cutoff)), index=individual.index),
            }
        )

    def setup_fault_t_do(self, individual: pd.DataFrame):
        """
        Return a binary sequence with each bit representing whether the current
        index is the time point at which the event of interest (i.e. a fault) occurred.

        N.B. This is rounded _up_ to the nearest multiple of `self.timesteps_per_observation`.
        That is, if the fault occurs at time 22, and `self.timesteps_per_observation == 5`, then
        `fault_t_do` will be 25.
        """

        fault = individual[~individual[self.status_column]]
        individual["fault_t_do"] = 0

        if not fault.empty:
            time_fault_observed = (
                max(0, np.ceil(fault["time"].min() / self.timesteps_per_observation) - 1)
            ) * self.timesteps_per_observation
            individual.loc[individual["time"] == time_fault_observed, "fault_t_do"] = 1

        assert sum(individual["fault_t_do"]) <= 1, f"Multiple fault times for\n{individual}"

        return pd.DataFrame({"fault_t_do": individual["fault_t_do"]})

    def setup_fault_time(self, individual: pd.DataFrame, perturbation: float = -0.001):
        """
        Return the time at which the event of interest (i.e. a fault) occurred.
        """
        fault = individual[~individual[self.status_column]]
        fault_time = (
            individual["time"].loc[fault.index[0]]
            if not fault.empty
            else (self.total_time + self.timesteps_per_observation)
        )
        return pd.DataFrame(
            {
                "fault_time": np.repeat(fault_time + perturbation, len(individual)),
            }
        )

    def preprocess_data(self):
        """
        Set up the treatment-specific columns in the data that are needed to estimate the hazard ratio.
        """

        self.df["trtrand"] = None  # treatment/control arm
        self.df["xo_t_do"] = None  # did the individual deviate from the treatment of interest here?
        self.df["eligible"] = self.df.eval(self.eligibility) if self.eligibility is not None else True

        # when did a fault occur?
        fault_time_df = self.df.groupby("id", sort=False)[[self.status_column, "time", "id"]].apply(
            self.setup_fault_time
        )

        assert len(fault_time_df) == len(self.df), "Fault times error"
        self.df["fault_time"] = fault_time_df["fault_time"].values

        assert (
            self.df.groupby("id", sort=False).apply(lambda x: len(set(x["fault_time"])) == 1).all()
        ), "Each individual must have a unique fault time."

        fault_t_do_df = self.df.groupby("id", sort=False)[["id", "time", self.status_column]].apply(
            self.setup_fault_t_do
        )
        assert len(fault_t_do_df) == len(self.df), "Fault t_do error"
        self.df["fault_t_do"] = fault_t_do_df["fault_t_do"].values

        living_runs = self.df.query("fault_time > 0").loc[
            (self.df["time"] % self.timesteps_per_observation == 0) & (self.df["time"] <= self.total_time)
        ]

        logging.debug("  Preprocessing groups")

        ctrl_time_0, ctrl_var_0, ctrl_val_0 = self.control_strategy[0]
        ctrl_time, ctrl_var, ctrl_val = min(
            set(map(tuple, self.control_strategy)).difference(map(tuple, self.treatment_strategy))
        )
        control_group = (
            living_runs.groupby("id", sort=False)
            .filter(lambda gp: len(gp.loc[(gp["time"] == ctrl_time) & (gp[ctrl_var] == ctrl_val)]) > 0)
            .groupby("id", sort=False)
            .filter(lambda gp: len(gp.loc[(gp["time"] == ctrl_time_0) & (gp[ctrl_var_0] == ctrl_val_0)]) > 0)
            .copy()
        )
        control_group["trtrand"] = 0
        ctrl_xo_t_do_df = control_group.groupby("id", sort=False).apply(
            self.setup_xo_t_do, strategy_assigned=self.control_strategy
        )
        control_group["xo_t_do"] = ctrl_xo_t_do_df["xo_t_do"].values
        control_group["old_id"] = control_group["id"]
        # control_group["id"] = ctrl_xo_t_do_df["id"].values
        control_group["id"] = [f"c-{id}" for id in control_group["id"]]
        assert not control_group["id"].isnull().any(), "Null control IDs"

        trt_time_0, trt_var_0, trt_val_0 = self.treatment_strategy[0]
        trt_time, trt_var, trt_val = min(
            set(map(tuple, self.treatment_strategy)).difference(map(tuple, self.control_strategy))
        )
        treatment_group = (
            living_runs.groupby("id", sort=False)
            .filter(lambda gp: len(gp.loc[(gp["time"] == trt_time) & (gp[trt_var] == trt_val)]) > 0)
            .groupby("id", sort=False)
            .filter(lambda gp: len(gp.loc[(gp["time"] == trt_time_0) & (gp[trt_var_0] == trt_val_0)]) > 0)
            .copy()
        )
        treatment_group["trtrand"] = 1
        trt_xo_t_do_df = treatment_group.groupby("id", sort=False).apply(
            self.setup_xo_t_do, strategy_assigned=self.treatment_strategy
        )
        treatment_group["xo_t_do"] = trt_xo_t_do_df["xo_t_do"].values
        treatment_group["old_id"] = treatment_group["id"]
        # treatment_group["id"] = trt_xo_t_do_df["id"].values
        treatment_group["id"] = [f"t-{id}" for id in treatment_group["id"]]
        assert not treatment_group["id"].isnull().any(), "Null treatment IDs"

        premature_failures = living_runs.groupby("id", sort=False).filter(lambda gp: gp["time"].max() < trt_time)
        logger.debug(
            f"{len(control_group.groupby('id'))} control individuals "
            f"{len(treatment_group.groupby('id'))} treatment individuals "
            f"{len(premature_failures.groupby('id'))} premature failures"
        )

        self.len_control_group = len(control_group.groupby("id"))
        self.len_treatment_group = len(treatment_group.groupby("id"))
        individuals = pd.concat([control_group, treatment_group])
        individuals = individuals.loc[
            (
                (
                    individuals["time"]
                    < np.ceil(individuals["fault_time"] / self.timesteps_per_observation)
                    * self.timesteps_per_observation
                )
                & (~individuals["xo_t_do"].isnull())
            )
        ]

        if len(individuals) == 0:
            raise ValueError("No individuals followed either strategy.")
        self.df = individuals.loc[
            individuals["time"]
            < np.ceil(individuals["fault_time"] / self.timesteps_per_observation) * self.timesteps_per_observation
        ].reset_index()
        logger.debug(f"{len(individuals.groupby('id'))} individuals")

        if len(self.df.loc[self.df["trtrand"] == 0]) == 0:
            raise ValueError(f"No individuals began the control strategy {self.control_strategy}")
        if len(self.df.loc[self.df["trtrand"] == 1]) == 0:
            raise ValueError(f"No individuals began the treatment strategy {self.treatment_strategy}")

    def estimate_hazard_ratio(self):
        """
        Estimate the hazard ratio.
        """

        if self.df["fault_t_do"].sum() == 0:
            raise ValueError("No recorded faults")

        preprocessed_data = self.df.copy()

        # Use logistic regression to predict switching given baseline covariates
        logger.debug("Use logistic regression to predict switching given baseline covariates")
        fit_bl_switch_c = smf.logit(self.fit_bl_switch_formula, data=self.df.loc[self.df.trtrand == 0]).fit(
            method="bfgs"
        )
        fit_bl_switch_t = smf.logit(self.fit_bl_switch_formula, data=self.df.loc[self.df.trtrand == 1]).fit(
            method="bfgs"
        )

        preprocessed_data.loc[preprocessed_data["trtrand"] == 0, "pxo1"] = fit_bl_switch_c.predict(
            self.df.loc[self.df.trtrand == 0]
        )
        preprocessed_data.loc[preprocessed_data["trtrand"] == 1, "pxo1"] = fit_bl_switch_t.predict(
            self.df.loc[self.df.trtrand == 1]
        )

        # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
        logger.debug(
            "Use logistic regression to predict switching given baseline and time-updated covariates (model S12)"
        )
        fit_bltd_switch_c = smf.logit(
            self.fit_bltd_switch_formula,
            data=self.df.loc[self.df.trtrand == 0],
        ).fit(method="bfgs")
        fit_bltd_switch_t = smf.logit(
            self.fit_bltd_switch_formula,
            data=self.df.loc[self.df.trtrand == 1],
        ).fit(method="bfgs")

        preprocessed_data.loc[preprocessed_data["trtrand"] == 0, "pxo2"] = fit_bltd_switch_c.predict(
            self.df.loc[self.df.trtrand == 0]
        )
        preprocessed_data.loc[preprocessed_data["trtrand"] == 1, "pxo2"] = fit_bltd_switch_t.predict(
            self.df.loc[self.df.trtrand == 1]
        )
        if (preprocessed_data["pxo2"] == 1).any():
            raise ValueError(
                "Probability of switching given baseline and time-varying confounders (pxo2) cannot be one."
            )

        # IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
        # Estimate the probabilities of remaining 'un-switched' and hence the weights
        logger.debug("Estimate the probabilities of remaining 'un-switched' and hence the weights")

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
            [(preprocessed_data["time"] + self.timesteps_per_observation), preprocessed_data["fault_time"]],
            axis=1,
        ).min(axis=1)

        assert (preprocessed_data["tin"] <= preprocessed_data["tout"]).all(), "Individuals left before joining."

        #  IPCW step 4: Use these weights in a weighted analysis of the outcome model
        # Estimate the KM graph and IPCW hazard ratio using Cox regression.
        logger.debug("Estimate the KM graph and IPCW hazard ratio using Cox regression.")
        cox_ph = CoxPHFitter(penalizer=0.2, alpha=self.alpha)
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
