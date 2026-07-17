"""This module contains the CausalTestCase class, a class that holds the information required for a causal test"""

import logging

import numpy as np
import pandas as pd

from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_effect import CausalEffect
from causal_testing.testing.causal_test_result import CausalTestResult, TestOutcome
from causal_testing.testing.data_adequacy import DataAdequacy

logger = logging.getLogger(__name__)


class CausalTestCase:
    # pylint: disable=too-many-instance-attributes
    """
    A CausalTestCase extends the information held in a BaseTestCase. As well as storing the treatment and outcome
    variables, a CausalTestCase stores the values of these variables. Also the outcome variable and value are
    specified. The goal of a CausalTestCase is to test whether the intervention made to the control via the treatment
    causes the model-under-test to produce the expected change.
    :param base_test_case: A BaseTestCase object consisting of a treatment variable, outcome variable and effect
    :param expected_causal_effect: The expected causal effect (Positive, Negative, No Effect).
    :param estimate_type: A string which denotes the type of estimate to return.
    :param estimator: An Estimator class object
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        base_test_case: BaseTestCase,
        expected_causal_effect: CausalEffect,
        estimate_type: str = "ate",
        estimator: type(Estimator) = None,
        name: str = None,
        query: str = None,
        skip: bool = False,
    ):
        self.base_test_case = base_test_case
        self.expected_causal_effect = expected_causal_effect
        self.outcome_variable = base_test_case.outcome_variable
        self.treatment_variable = base_test_case.treatment_variable
        self.estimate_type = estimate_type
        self.estimator = estimator
        self.effect = base_test_case.effect
        self.result = None
        self.name = name
        self.query = query
        self.skip = skip

    def measure_adequacy(
        self,
        df: pd.DataFrame,
        bootstrap_size: int = 100,
        group_by: str = None,
    ) -> DataAdequacy:
        """
        Calculate the adequacy measurement, and populate the data_adequacy field.
        :param df: The original dataset to use.
        :param bootstrap_size: The number of bootstrap samples to use. (Defaults to 100)
        :param group_by: For IPCWEstimator - the "id" column to ensure that entire individuals are sampled rather than
        random rows.
        """
        results = []
        outcomes = []
        for i in range(bootstrap_size):
            if group_by is not None:
                ids = pd.Series(df[group_by].unique())
                ids = ids.sample(len(ids), replace=True, random_state=i)
                df = df[df[group_by].isin(ids)]
            else:
                df = df.sample(len(df), replace=True, random_state=i)
            try:
                effect_estimate = self.estimate_effect(df)
                outcomes.append(self.expected_causal_effect.apply(effect_estimate))
                results.append(effect_estimate.to_df())
            # Could get a variety of exceptions here due to insufficient/badly formed data
            # We don't want these to stop execution
            except Exception:  # pylint: disable=W0718
                pass

        results = pd.concat(results)

        results["var"] = results.index
        results["passed"] = outcomes

        return DataAdequacy(
            results=results,
            kurtosis=results.groupby("var")["effect_estimate"].apply(lambda x: x.kurtosis()),
            passing=sum(filter(lambda x: x is not None, outcomes)),
            successful=sum(x is not None for x in outcomes),
        )

    def execute_test(
        self,
        df: pd.DataFrame,
        adequacy: bool = False,
        suppress_estimation_errors: bool = False,
        bootstrap_size: int = 100,
        group_by: str = None,
    ):
        """
        Execute a causal test case.

        :param df: The data to use.
        :param adequacy: Set to True to calculate the causal test adequacy associated with the effect estimate.
        :param suppress_estimation_errors: Set to True to suppress estimation errors. (Defaults to False)
        :param bootstrap_size: The number of bootstrap samples to use. (Defaults to 100)
        :param group_by: For IPCWEstimator - the "id" column to ensure that entire individuals are sampled rather than
        random rows.
        :return causal_test_result: A CausalTestResult for the executed causal test case.
        """
        if not self.skip:
            try:
                effect_estimate = self.estimate_effect(df=df)
                self.result = CausalTestResult(
                    effect_estimate=effect_estimate,
                    outcome=(
                        TestOutcome.PASS
                        if self.expected_causal_effect.apply(effect_estimate=effect_estimate)
                        else TestOutcome.FAIL
                    ),
                    adequacy=(
                        self.measure_adequacy(df=df, bootstrap_size=bootstrap_size, group_by=group_by)
                        if adequacy
                        else None
                    ),
                )
            except (np.linalg.LinAlgError, ValueError) as e:
                if not suppress_estimation_errors:
                    raise e
                self.result = CausalTestResult(
                    effect_estimate=None, outcome=TestOutcome.INESTIMABLE, error_message=str(e)
                )

    def estimate_effect(self, df: pd.DataFrame) -> CausalTestResult:
        """
        Execute a causal test case and return the causal test result.

        :param df: The data to use.
        :return causal_test_result: A CausalTestResult for the executed causal test case.
        """
        if self.query:
            df = df.query(self.query)
        if not hasattr(self.estimator, f"estimate_{self.estimate_type}"):
            raise AttributeError(f"{self.estimator.__class__} has no {self.estimate_type} method.")
        estimate_effect = getattr(self.estimator, f"estimate_{self.estimate_type}")
        return estimate_effect(df)

    def __str__(self):
        treatment_config = {self.treatment_variable.name: self.estimator.treatment_value}
        control_config = {self.treatment_variable.name: self.estimator.control_value}
        outcome_variable = {self.outcome_variable.name}
        return (
            f"Running {treatment_config} instead of {control_config} should cause the following "
            f"changes to {outcome_variable}: {self.expected_causal_effect}."
        )
