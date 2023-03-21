# pylint: disable=too-few-public-methods
"""This module contains the CausalTestOutcome abstract class, as well as the concrete extension classes:
ExactValue, Positive, Negative, SomeEffect, NoEffect"""

from abc import ABC, abstractmethod
import numpy as np

from causal_testing.testing.causal_test_result import CausalTestResult


class CausalTestOutcome(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(self, res: CausalTestResult) -> bool:
        """Abstract apply method that should return a bool representing if the result meets the outcome
        :param res: CausalTestResult to be checked
        :return: Bool that is true if outcome is met
        """

    def __str__(self) -> str:
        return type(self).__name__


class SomeEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should not be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type == "ate":
            return (0 < res.ci_low() < res.ci_high()) or (res.ci_low() < res.ci_high() < 0)
        if res.test_value.type == "risk_ratio":
            return (1 < res.ci_low() < res.ci_high()) or (res.ci_low() < res.ci_high() < 1)
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")


class NoEffect(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type == "ate":
            return (res.ci_low() < 0 < res.ci_high()) or (abs(res.test_value.value) < 1e-10)
        if res.test_value.type == "risk_ratio":
            return (res.ci_low() < 1 < res.ci_high()) or np.isclose(res.test_value.value, 1.0, atol=1e-10)
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")


class ExactValue(SomeEffect):
    """An extension of TestOutcome representing that the expected causal effect should be a specific value."""

    def __init__(self, value: float, tolerance: float = None):
        self.value = value
        if tolerance is None:
            self.tolerance = value * 0.05
        else:
            self.tolerance = tolerance

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_valid():
            return super().apply(res) and np.isclose(res.test_value.value, self.value, atol=self.tolerance)
        return np.isclose(res.test_value.value, self.value, atol=self.tolerance)

    def __str__(self):
        return f"ExactValue: {self.value}±{self.tolerance}"


class Positive(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_valid() and not super().apply(res):
            return False
        if res.test_value.type == "ate":
            return res.test_value.value > 0
        if res.test_value.type == "risk_ratio":
            return res.test_value.value > 1
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")


class Negative(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be negative."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_valid() and not super().apply(res):
            return False
        if res.test_value.type == "ate":
            return res.test_value.value < 0
        if res.test_value.type == "risk_ratio":
            return res.test_value.value < 1
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this TestOutcome")
