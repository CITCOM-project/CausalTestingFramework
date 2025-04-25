# pylint: disable=too-few-public-methods
"""This module contains the CausalEffect abstract class, as well as the concrete extension classes:
ExactValue, Positive, Negative, SomeEffect, NoEffect"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np

from causal_testing.testing.causal_test_result import CausalTestResult


class CausalEffect(ABC):
    """An abstract class representing an expected causal effect."""

    @abstractmethod
    def apply(self, res: CausalTestResult) -> bool:
        """Abstract apply method that should return a bool representing if the result meets the outcome
        :param res: CausalTestResult to be checked
        :return: Bool that is true if outcome is met
        """

    def __str__(self) -> str:
        return type(self).__name__


class SomeEffect(CausalEffect):
    """An extension of CausalEffect representing that the expected causal effect should not be zero."""

    def apply(self, res: CausalTestResult) -> bool:
        if res.ci_low() is None or res.ci_high() is None:
            return None
        if res.test_value.type in ("risk_ratio", "hazard_ratio", "unit_odds_ratio"):
            return any(
                1 < ci_low < ci_high or ci_low < ci_high < 1 for ci_low, ci_high in zip(res.ci_low(), res.ci_high())
            )
        if res.test_value.type in ("coefficient", "ate"):
            return any(
                0 < ci_low < ci_high or ci_low < ci_high < 0 for ci_low, ci_high in zip(res.ci_low(), res.ci_high())
            )

        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this CausalEffect")


class NoEffect(CausalEffect):
    """An extension of CausalEffect representing that the expected causal effect should be zero."""

    def __init__(self, atol: float = 1e-10, ctol: float = 0.05):
        """
        :param atol: Arithmetic tolerance. The test will pass if the absolute value of the causal effect is less than
                     atol.
        :param ctol: Categorical tolerance. The test will pass if this proportion of categories pass.
        """
        self.atol = atol
        self.ctol = ctol

    def apply(self, res: CausalTestResult) -> bool:
        if res.test_value.type in ("risk_ratio", "hazard_ratio", "unit_odds_ratio"):
            return any(
                ci_low < 1 < ci_high or np.isclose(value, 1.0, atol=self.atol)
                for ci_low, ci_high, value in zip(res.ci_low(), res.ci_high(), res.test_value.value)
            )
        if res.test_value.type in ("coefficient", "ate"):
            value = res.test_value.value if isinstance(res.ci_high(), Iterable) else [res.test_value.value]
            return (
                sum(
                    not ((ci_low < 0 < ci_high) or abs(v) < self.atol)
                    for ci_low, ci_high, v in zip(res.ci_low(), res.ci_high(), value)
                )
                / len(value)
                < self.ctol
            )

        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this CausalEffect")


class ExactValue(CausalEffect):
    """An extension of CausalEffect representing that the expected causal effect should be a specific value."""

    def __init__(self, value: float, atol: float = None, ci_low: float = None, ci_high: float = None):
        if (ci_low is not None) ^ (ci_high is not None):
            raise ValueError("If specifying confidence intervals, must specify `ci_low` and `ci_high` parameters.")
        if atol is not None and atol < 0:
            raise ValueError("Tolerance must be an absolute (positive) value.")

        self.value = value
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.atol = atol if atol is not None else abs(value * 0.05)

        if self.ci_low is not None and self.ci_high is not None:
            if not self.ci_low <= self.value <= self.ci_high:
                raise ValueError("Specified value falls outside the specified confidence intervals.")
            if self.value - self.atol < self.ci_low or self.value + self.atol > self.ci_high:
                raise ValueError(
                    "Arithmetic tolerance falls outside the confidence intervals."
                    "Try specifying a smaller value of atol."
                )

    def apply(self, res: CausalTestResult) -> bool:
        close = np.isclose(res.test_value.value, self.value, atol=self.atol)
        if res.ci_valid() and self.ci_low is not None and self.ci_high is not None:
            return all(
                close and self.ci_low <= ci_low and self.ci_high >= ci_high
                for ci_low, ci_high in zip(res.ci_low(), res.ci_high())
            )
        return close

    def __str__(self):
        return f"ExactValue: {self.value}±{self.atol}"


class Positive(SomeEffect):
    """An extension of CausalEffect representing that the expected causal effect should be positive.
    Currently only single values are supported for the test value"""

    def apply(self, res: CausalTestResult) -> bool:
        if len(res.test_value.value) > 1:
            raise ValueError("Positive Effects are currently only supported on single float datatypes")
        if res.test_value.type in {"ate", "coefficient"}:
            return bool(res.test_value.value[0] > 0)
        if res.test_value.type == "risk_ratio":
            return bool(res.test_value.value[0] > 1)
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this CausalEffect")


class Negative(SomeEffect):
    """An extension of CausalEffect representing that the expected causal effect should be negative.
    Currently only single values are supported for the test value"""

    def apply(self, res: CausalTestResult) -> bool:
        if len(res.test_value.value) > 1:
            raise ValueError("Negative Effects are currently only supported on single float datatypes")
        if res.test_value.type in {"ate", "coefficient"}:
            return bool(res.test_value.value[0] < 0)
        if res.test_value.type == "risk_ratio":
            return bool(res.test_value.value[0] < 1)
        # Dead code but necessary for pylint
        raise ValueError(f"Test Value type {res.test_value.type} is not valid for this CausalEffect")
