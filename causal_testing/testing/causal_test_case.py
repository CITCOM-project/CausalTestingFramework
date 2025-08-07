"""This module contains the CausalTestCase class, a class that holds the information required for a causal test"""

import logging

from causal_testing.testing.causal_effect import CausalEffect
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.testing.causal_test_result import CausalTestResult


logger = logging.getLogger(__name__)


class CausalTestCase:
    # pylint: disable=too-many-instance-attributes
    """
    A CausalTestCase extends the information held in a BaseTestCase. As well as storing the treatment and outcome
    variables, a CausalTestCase stores the values of these variables. Also the outcome variable and value are
    specified. The goal of a CausalTestCase is to test whether the intervention made to the control via the treatment
               causes the model-under-test to produce the expected change.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        base_test_case: BaseTestCase,
        expected_causal_effect: CausalEffect,
        estimate_type: str = "ate",
        estimate_params: dict = None,
        estimator: type(Estimator) = None,
    ):
        """
        :param base_test_case: A BaseTestCase object consisting of a treatment variable, outcome variable and effect
        :param expected_causal_effect: The expected causal effect (Positive, Negative, No Effect).
        :param estimate_type: A string which denotes the type of estimate to return.
        :param estimator: An Estimator class object
        """
        self.base_test_case = base_test_case
        self.expected_causal_effect = expected_causal_effect
        self.outcome_variable = base_test_case.outcome_variable
        self.treatment_variable = base_test_case.treatment_variable
        self.estimate_type = estimate_type
        self.estimator = estimator
        if estimate_params is None:
            self.estimate_params = {}

        else:
            self.estimate_params = estimate_params

        self.effect = base_test_case.effect

    def execute_test(self, estimator: type(Estimator) = None) -> CausalTestResult:
        """
        Execute a causal test case and return the causal test result.
        :param estimator: An alternative estimator. Defaults to `self.estimator`. This parameter is useful when you want
        to execute a test with different data or a different equational form, but don't want to redefine the whole test
        case.

        :return causal_test_result: A CausalTestResult for the executed causal test case.
        """

        if estimator is None:
            estimator = self.estimator

        if not hasattr(estimator, f"estimate_{self.estimate_type}"):
            raise AttributeError(f"{estimator.__class__} has no {self.estimate_type} method.")
        estimate_effect = getattr(estimator, f"estimate_{self.estimate_type}")
        effect_estimate = estimate_effect(**self.estimate_params)
        return CausalTestResult(
            estimator=estimator,
            effect_estimate=effect_estimate,
        )

    def __str__(self):
        treatment_config = {self.treatment_variable.name: self.estimator.treatment_value}
        control_config = {self.treatment_variable.name: self.estimator.control_value}
        outcome_variable = {self.outcome_variable.name}
        return (
            f"Running {treatment_config} instead of {control_config} should cause the following "
            f"changes to {outcome_variable}: {self.expected_causal_effect}."
        )
