"""This module contains the IPCWEstimator class, for estimating the time to a particular event"""

import logging
from math import ceil

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

from causal_testing.specification.capabilities import TreatmentSequence, Capability
from causal_testing.estimation.abstract_estimator import Estimator

logger = logging.getLogger(__name__)


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
