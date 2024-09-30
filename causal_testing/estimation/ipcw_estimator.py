"""This module contains the IPCWEstimator class, for estimating the time to a particular event"""

import logging
from math import ceil
from typing import Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

from causal_testing.estimation.abstract_estimator import Estimator

logger = logging.getLogger(__name__)

debug_id = "data-50/batch_run_16/00221634_10.csv"


class IPCWEstimator(Estimator):
    """
    Class to perform Inverse Probability of Censoring Weighting (IPCW) estimation
    for sequences of treatments over time-varying data.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        df: pd.DataFrame,
        timesteps_per_observation: int,
        control_strategy: list[tuple[int, str, Any]],
        treatment_strategy: list[tuple[int, str, Any]],
        outcome: str,
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
            [var for _, var, _ in treatment_strategy],
            [val for _, _, val in treatment_strategy],
            [val for _, _, val in control_strategy],
            None,
            outcome,
            df,
            None,
            alpha=alpha,
            query="",
        )
        self.timesteps_per_observation = timesteps_per_observation
        self.control_strategy = control_strategy
        self.treatment_strategy = treatment_strategy
        self.outcome = outcome
        self.status_column = status_column
        self.fit_bl_switch_formula = fit_bl_switch_formula
        self.fit_bltd_switch_formula = fit_bltd_switch_formula
        self.eligibility = eligibility
        self.df = df.sort_values(["id", "time"])

        if total_time is None:
            total_time = (
                max(len(self.control_strategy), len(self.treatment_strategy)) + 1
            ) * self.timesteps_per_observation
        self.total_time = total_time
        print("PREPROCESSING")
        self.preprocess_data()
        print("PREPROCESSED")

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

        strategy_assigned = {t: (var, val) for t, var, val in strategy_assigned}
        strategy_followed = {t: (var, val) for t, var, val in strategy_followed}

        # fill in the gaps
        for time in eligible.index:
            if time not in strategy_assigned:
                strategy_assigned[time] = (-1, -1)
            if time not in strategy_followed:
                strategy_followed[time] = (-1, -1)

        strategy_assigned = sorted(
            [(t, var, val) for t, (var, val) in strategy_assigned.items() if t in eligible.index]
        )
        strategy_followed = sorted(
            [(t, var, val) for t, (var, val) in strategy_followed.items() if t in eligible.index]
        )

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
        index is the time point at which the event of interest (i.e. a fault) occurred.

        N.B. This is rounded _up_ to the nearest multiple of `self.timesteps_per_observation`.
        That is, if the fault occurs at time 22, and `self.timesteps_per_observation == 5`, then
        `fault_t_do` will be 25.
        """

        fault = individual[~individual[self.status_column]]
        individual["fault_t_do"] = 0

        if not fault.empty:
            time_fault_observed = (
                max(0, ceil(fault["time"].min() / self.timesteps_per_observation) - 1)
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
        ), f"Each individual must have a unique fault time."

        fault_t_do_df = self.df.groupby("id", sort=False)[["id", "time", self.status_column]].apply(
            self.setup_fault_t_do
        )
        assert len(fault_t_do_df) == len(self.df), "Fault t_do error"
        self.df["fault_t_do"] = fault_t_do_df["fault_t_do"].values

        living_runs = self.df.query("fault_time > 0").loc[
            (self.df["time"] % self.timesteps_per_observation == 0) & (self.df["time"] <= self.total_time)
        ]

        individuals = []
        new_id = 0
        logging.debug("  Preprocessing groups")

        for id, individual in tqdm(living_runs.groupby("id", sort=False)):
            assert sum(individual["fault_t_do"]) <= 1, (
                f"Error initialising fault_t_do for individual\n"
                f"{individual[['id', 'time', self.status_column, 'fault_time', 'fault_t_do']]}\n"
                f"with fault at {individual.fault_time.iloc[0]}"
            )

            strategy_followed = [
                [t, var, individual.loc[individual["time"] == t, var].values[0]]
                for t, var, val in self.treatment_strategy
                if t in individual["time"].values
            ]

            # Control flow:
            # Individuals that start off in both arms, need cloning (hence incrementing the ID within the if statement)
            # Individuals that don't start off in either arm are left out
            for inx, strategy_assigned in [(0, self.control_strategy), (1, self.treatment_strategy)]:
                if (
                    len(strategy_followed) > 0
                    and strategy_assigned[0] == strategy_followed[0]
                    and individual.eligible.iloc[0]
                ):
                    individual["old_id"] = individual["id"]
                    individual["id"] = new_id
                    new_id += 1
                    individual["trtrand"] = inx
                    individual["xo_t_do"] = self.setup_xo_t_do(
                        strategy_assigned, strategy_followed, individual["eligible"]
                    )
                    individuals.append(
                        individual.loc[
                            individual["time"]
                            < ceil(individual["fault_time"].iloc[0] / self.timesteps_per_observation)
                            * self.timesteps_per_observation
                        ].copy()
                    )
        if len(individuals) == 0:
            raise ValueError("No individuals followed either strategy.")
        self.df = pd.concat(individuals)
        print(len(individuals), "individuals")

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
        print("Use logistic regression to predict switching given baseline covariates")
        fit_bl_switch = smf.logit(self.fit_bl_switch_formula, data=self.df).fit()

        preprocessed_data["pxo1"] = fit_bl_switch.predict(preprocessed_data)

        # Use logistic regression to predict switching given baseline and time-updated covariates (model S12)
        print("Use logistic regression to predict switching given baseline and time-updated covariates (model S12)")
        fit_bltd_switch = smf.logit(
            self.fit_bltd_switch_formula,
            data=self.df,
        ).fit()

        preprocessed_data["pxo2"] = fit_bltd_switch.predict(preprocessed_data)
        if (preprocessed_data["pxo2"] == 1).any():
            raise ValueError(
                "Probability of switching given baseline and time-varying confounders (pxo2) cannot be one."
            )

        # IPCW step 3: For each individual at each time, compute the inverse probability of remaining uncensored
        # Estimate the probabilities of remaining ‘un-switched’ and hence the weights
        print("Estimate the probabilities of remaining ‘un-switched’ and hence the weights")

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

        assert (preprocessed_data["tin"] <= preprocessed_data["tout"]).all(), (
            f"Left before joining\n"
            f"{preprocessed_data.loc[preprocessed_data['tin'] >= preprocessed_data['tout'], ['id', 'time', 'fault_time', 'tin', 'tout']]}"
        )

        preprocessed_data.pop("old_id")
        assert (
            not np.isinf(preprocessed_data[[col for col in preprocessed_data if preprocessed_data.dtypes[col] != bool]])
            .any()
            .any()
        ), "Infinity not allowed."

        #  IPCW step 4: Use these weights in a weighted analysis of the outcome model
        # Estimate the KM graph and IPCW hazard ratio using Cox regression.
        print("Estimate the KM graph and IPCW hazard ratio using Cox regression.")
        cox_ph = CoxPHFitter()
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
