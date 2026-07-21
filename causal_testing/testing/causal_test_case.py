"""This module contains the CausalTestCase class, a class that holds the information required for a causal test"""

import logging

import numpy as np
import pandas as pd

from causal_testing.estimation.abstract_estimator import Estimator
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
    :param effect_measure: A string which denotes the type of estimate to return.
    :param estimator: An Estimator class object
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment_variable: str,
        outcome_variable: str,
        expected_causal_effect: CausalEffect,
        effect_measure: str,
        estimator: type(Estimator) = None,
        name: str = None,
        query: str = None,
        skip: bool = False,
    ):
        self.treatment_variable = treatment_variable
        self.outcome_variable = outcome_variable
        self.expected_causal_effect = expected_causal_effect
        self.effect_measure = effect_measure
        self.estimator = estimator
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
                sample_df = df[df[group_by].isin(ids)]
            else:
                sample_df = df.sample(len(df), replace=True, random_state=i)
            try:
                effect_estimate = self.estimate_effect(sample_df)
                outcomes.append(self.expected_causal_effect.apply(effect_estimate))
                results.append(
                    effect_estimate.to_df().assign(
                        test_index=i, passed=self.expected_causal_effect.apply(effect_estimate)
                    )
                )
            # Could get a variety of exceptions here due to insufficient/badly formed data in the sample
            # We don't want these to stop execution
            except Exception:  # pylint: disable=W0718
                outcomes.append(None)

        results = pd.concat(results)

        results["var"] = results.index

        return DataAdequacy(
            results=results,
            kurtosis=results.groupby("var")["effect_estimate"].apply(lambda x: x.kurtosis()),
            passing=int(sum(filter(lambda x: x is not None, outcomes))),
            successful=int(sum(x is not None for x in outcomes)),
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
        if not hasattr(self.estimator, f"estimate_{self.effect_measure}"):
            raise AttributeError(f"{self.estimator.__class__} has no {self.effect_measure} method.")
        estimate_effect = getattr(self.estimator, f"estimate_{self.effect_measure}")
        return estimate_effect(df)

    def to_dict(self) -> dict:
        """
        Convert the test case to a python dictionary for easy serialisation as JSON.

        :returns: A JSON serialisable dict representing the test case.
        """
        test_case = {
            "name": self.name,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "skip": self.skip,
            "effect_measure": self.effect_measure,
            "query": self.query,
        }

        for label, attribute in [
            ("expected_effect", self.expected_causal_effect),
            ("estimator", self.estimator),
            ("result", self.result),
        ]:
            if attribute is not None:
                test_case[label] = attribute.to_dict()

        return test_case
