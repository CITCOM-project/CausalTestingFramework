"""This module contains the CausalTestResult class, which is a container for the results of a causal test, and the
TestValue dataclass.
"""
from typing import Any
from dataclasses import dataclass

from causal_testing.testing.estimators import Estimator
from causal_testing.specification.variable import Variable


@dataclass
class TestValue:
    """A dataclass to hold both the type and value of a causal test result"""

    type: str
    value: float


class CausalTestResult:
    """A container to hold the results of a causal test case. Every causal test case provides a point estimate of
    the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
    confidence intervals."""

    def __init__(
        self,
        estimator: Estimator,
        test_value: TestValue,
        confidence_intervals: [float, float] = None,
        effect_modifier_configuration: {Variable: Any} = None,
    ):
        self.estimator = estimator
        if estimator.adjustment_set:
            self.adjustment_set = estimator.adjustment_set
        else:
            self.adjustment_set = set()
        self.test_value = test_value
        self.confidence_intervals = confidence_intervals

        if effect_modifier_configuration is not None:
            self.effect_modifier_configuration = effect_modifier_configuration
        else:
            self.effect_modifier_configuration = {}

    def __str__(self):
        base_str = (
            f"Causal Test Result\n==============\n"
            f"Treatment: {self.estimator.treatment[0]}\n"
            f"Control value: {self.estimator.control_value}\n"
            f"Treatment value: {self.estimator.treatment_value}\n"
            f"Outcome: {self.estimator.outcome[0]}\n"
            f"Adjustment set: {self.adjustment_set}\n"
            f"{self.test_value.type}: {self.test_value.value}\n"
        )
        confidence_str = ""
        if self.confidence_intervals:
            confidence_str += f"Confidence intervals: {self.confidence_intervals}\n"
        return base_str + confidence_str

    def to_dict(self):
        """Return result contents as a dictionary
        :return: Dictionary containing contents of causal_test_result
        """
        base_dict = {
            "treatment": self.estimator.treatment[0],
            "control_value": self.estimator.control_value,
            "treatment_value": self.estimator.treatment_value,
            "outcome": self.estimator.outcome[0],
            "adjustment_set": self.adjustment_set,
            "test_value": self.test_value,
        }
        if self.confidence_intervals and all(self.confidence_intervals):
            base_dict["ci_low"] = min(self.confidence_intervals)
            base_dict["ci_high"] = max(self.confidence_intervals)
        return base_dict

    def ci_low(self):
        """Return the lower bracket of the confidence intervals."""
        if self.confidence_intervals and all(self.confidence_intervals):
            return min(self.confidence_intervals)
        return None

    def ci_high(self):
        """Return the higher bracket of the confidence intervals."""
        if self.confidence_intervals and all(self.confidence_intervals):
            return max(self.confidence_intervals)
        return None

    def ci_valid(self) -> bool:
        """Return whether or not the result has valid confidence invervals"""
        return self.ci_low() and self.ci_high()

    def summary(self):
        """Summarise the causal test result as an intuitive sentence."""
        print(
            f"The causal effect of changing {self.estimator.treatment[0]} = {self.estimator.control_value} to "
            f"{self.estimator.treatment[0]}' = {self.estimator.treatment_value} is {self.test_value.value}"
            f"(95% confidence intervals: {self.confidence_intervals})."
        )
