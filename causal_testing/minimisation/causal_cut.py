"""
This module implements the Causal Cut tool to minimise test sequences for cyberphysical systems.
See https://doi.org/10.1145/3816435 for further details.
"""

import logging
from collections.abc import Callable

import pandas as pd

from causal_testing.estimation.ipcw_estimator import IPCWEstimator
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_effect import SomeEffect
from causal_testing.testing.causal_test_case import CausalTestCase


class CausalCut:
    """
    Main class for minimising intervention sequences.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self, dag: CausalDAG, df: pd.DataFrame, safe_ranges: pd.DataFrame, reproduce_fault: Callable, ci_alpha=0.05
    ):
        self.dag = dag
        self.df = df
        self.safe_ranges = safe_ranges
        self.ci_alpha = ci_alpha
        self.reproduce_fault = reproduce_fault

    def estimate_intervention_causality(  # pylint: disable=too-many-arguments
        self,
        interventions: list[tuple[int, str, int]],
        outcome_variable: str,
        total_time: int,
        start_time: int = 0,
        background_confounders: list[str] = None,
        timesteps_per_intervention: int = 1,
    ) -> list[CausalTestCase]:
        """
        For each intervention, generate a causal test case to compare the original treatment strategy against the one
        with that intervention negated. This gives the causal contribution of each intervention to the observed failure.

        :param interventions: The list of interventions to prune, of the form [(time, variable, value)].
        :param outcome_variable: The name of the outcome variable.
        :param total_time: The maximum number of time steps that a test case can run for.
        :param start_time: The time at which the test case begins. (Defaults to 0)
        :param background_confounders: The names of the non-time-varying confounders. These variables remain constant
                                       throughout the whole test execution.
        :param timesteps_per_intervention: The number of time steps each intervention takes. (Defaults to 1)
        """
        background_confounders = background_confounders if background_confounders is not None else []

        interventions = list(filter(lambda x: start_time <= x[0] <= total_time, interventions))

        lo = self.safe_ranges.loc[outcome_variable, "low"]
        hi = self.safe_ranges.loc[outcome_variable, "high"]

        logging.debug(f"CONTROL STRATEGY   {interventions}")

        if not (~self.df[outcome_variable].between(lo, hi)).any():
            raise ValueError(
                f"No faults with {outcome_variable}. Cannot perform estimation.\n"
                f"Observed range [{self.df[outcome_variable].min()}, {self.df[outcome_variable].max()}].\n"
                f"Safe range {self.safe_ranges[outcome_variable]}"
            )
        if self.df[outcome_variable].between(lo, hi).all():
            raise ValueError(
                f"All faults with {outcome_variable}. Cannot perform estimation.\n"
                f"Observed range [{self.df[outcome_variable].min()}, {self.df[outcome_variable].max()}].\n"
                f"Safe range {self.safe_ranges[outcome_variable]}"
            )
        if any(var not in self.df for _, var, _ in interventions):
            raise ValueError("Missing data for control strategy")
        if any(var not in self.dag.nodes for _, var, _ in interventions):
            missing = [var for _, var, _ in interventions if var not in self.dag.nodes]
            raise ValueError(f"Missing nodes {missing} for control_strategy. Valid nodes {self.dag.nodes}")

        causal_tests = {}

        for i, (time, variable, value) in enumerate(interventions):
            intervention = (time, variable, value)
            logging.debug(f"Event {i}/{len(interventions)}")
            # Treatment strategy is the same, but with one intervention negated
            # i.e. we examine the counterfactual "What if we had not done that?"
            treatment_strategy = [x[:] for x in interventions]
            treatment_strategy[i][2] = int(not value)

            logging.debug(f"  TREATMENT STRATEGY {treatment_strategy}")
            logging.debug(f"  outcome_variable {outcome_variable}")
            logging.debug(f"  SAFE RANGE {lo} {hi}")

            neighbours = list(self.dag.predecessors(variable))
            neighbours += list(self.dag.successors(variable))

            if len(neighbours) == 0:
                raise ValueError(f"No neighbours for node {variable}.")

            if "time" not in background_confounders:
                background_confounders.append("time")
            fit_bl_switch_formula = f"xo_t_do ~ {' + '.join(background_confounders)}"
            self.df["within_safe_range"] = self.df[outcome_variable].between(lo, hi)

            causal_test_case = CausalTestCase(
                expected_causal_effect=SomeEffect(),
                effect_measure="hazard_ratio",
                estimator=IPCWEstimator(
                    timesteps_per_intervention,
                    interventions,
                    treatment_strategy,
                    outcome_variable,
                    "within_safe_range",
                    fit_bl_switch_formula=fit_bl_switch_formula,
                    fit_bltd_switch_formula=f"{fit_bl_switch_formula} + {' + '.join(neighbours)}",
                    eligibility=None,
                    alpha=self.ci_alpha,
                    total_time=total_time,
                ),
            )
            causal_test_case.execute_test(
                df=self.df.loc[self.df["time"].between(start_time, total_time)],
                suppress_estimation_errors=True,
            )
            causal_tests[intervention] = causal_test_case

        return causal_tests

    def prune_interventions(
        self,
        causal_tests: list[CausalTestCase],
        greedy_minimise: bool = False,
        **kwargs,
    ):
        """
        Remove interventions not estimated as having a significant causal effect on the observed failure.
        Run the attack and check if the interventions estimated significant lead to a fault.
        Add back interventions until the failure manifests.

        :param causal_tests: The causal tests (with results) from the estimation phase.
        :param greedy_minimise: Whether to apply additional greedy minimisation. (Defaults to False)
        :param kwargs: Keyword arguments for `self.reproduce_fault`.
        """

        # Phase 1 - prune interventions not estimated to be causally significant
        treatment_strategies = pd.json_normalize(
            [test.to_dict() | {"intervention": intervention} for intervention, test in causal_tests.items()]
        )
        treatment_strategies["time"] = [time for time, _, _ in treatment_strategies["intervention"]]

        treatment_strategies.to_csv("/tmp/treatment_strategies.csv")

        if "result.ci_low.trtrand" in treatment_strategies and "result.ci_high.trtrand" in treatment_strategies:
            treatment_strategies["result.ci_low.trtrand"] = treatment_strategies["result.ci_low.trtrand"]
            treatment_strategies["result.ci_high.trtrand"] = treatment_strategies["result.ci_high.trtrand"]
            treatment_strategies["significant"] = (treatment_strategies["result.ci_low.trtrand"] > 1) | (
                treatment_strategies["result.ci_high.trtrand"] < 1
            )
            treatment_strategies = treatment_strategies.loc[~treatment_strategies["significant"]]
            treatment_strategies["below_1"] = (1 - treatment_strategies["result.ci_low.trtrand"]) / (
                treatment_strategies["result.ci_high.trtrand"] - treatment_strategies["result.ci_low.trtrand"]
            )
            treatment_strategies["above_1"] = (treatment_strategies["result.ci_high.trtrand"] - 1) / (
                treatment_strategies["result.ci_high.trtrand"] - treatment_strategies["result.ci_low.trtrand"]
            )
            treatment_strategies["rank"] = treatment_strategies[["below_1", "above_1"]].min(axis=1)
            # Sort by rank (low -> high), then last -> first
            treatment_strategies.sort_values(["rank", "time"], inplace=True, ascending=[True, False])
        else:
            treatment_strategies.sort_values(["time"], inplace=True, ascending=False)

        interventions = treatment_strategies.loc[treatment_strategies["result.passed"], "intervention"].to_list()

        treatment_strategies.to_csv("/tmp/treatment_strategies_sorted.csv")

        estimated_interventions = list(interventions)

        # Check whether the pruned test yields the original fault
        still_fault = self.reproduce_fault(interventions=interventions, **kwargs)

        # Phase 2 - add interventions back to the test until the original fault is yielded
        interventions_to_add = treatment_strategies.loc[
            ~treatment_strategies["result.passed"], "intervention"
        ].to_list()
        while not still_fault and interventions_to_add:
            next_intervention = interventions_to_add.pop(0)
            if next_intervention in interventions:
                continue
            interventions.append(next_intervention)
            still_fault = self.reproduce_fault(interventions=interventions, **kwargs)
        interventions.sort()

        # Apply the greedy heuristic to the tool-minimised trace
        if greedy_minimise:
            for intervention in sorted(estimated_interventions):
                if self.reproduce_fault(interventions=[i for i in interventions if i != intervention], **kwargs):
                    interventions.remove(intervention)

        return interventions

    def minimise_test(  # pylint: disable=too-many-arguments
        self,
        interventions: list[tuple[int, str, int]],
        outcome_variable: str,
        total_time: int,
        start_time: int = 0,
        background_confounders: list[str] = None,
        timesteps_per_intervention: int = 1,
        greedy_minimise: bool = False,
        **kwargs,
    ) -> list[tuple[int, str, int]]:
        """
        Search for a subset of the supplied interventions that still yields the originally observed failure.
        :param interventions: The list of interventions to prune, of the form [(time, variable, value)].
        :param outcome_variable: The name of the outcome variable.
        :param total_time: The maximum number of time steps that a test case can run for.
        :param start_time: The time at which the test case begins. (Defaults to 0)
        :param background_confounders: The names of the non-time-varying confounders. These variables remain constant
                                       throughout the whole test execution.
        :param timesteps_per_intervention: The number of time steps each intervention takes. (Defaults to 1)
        :param greedy_minimise: Whether to apply additional greedy minimisation. (Defaults to False)
        :param kwargs: Keyword arguments for `self.reproduce_fault`.
        """
        causal_tests = self.estimate_intervention_causality(
            interventions=interventions,
            outcome_variable=outcome_variable,
            total_time=total_time,
            start_time=start_time,
            background_confounders=background_confounders,
            timesteps_per_intervention=timesteps_per_intervention,
        )

        return self.prune_interventions(
            causal_tests=causal_tests,
            greedy_minimise=greedy_minimise,
            **kwargs,
        )
