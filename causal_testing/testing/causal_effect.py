# pylint: disable=too-few-public-methods
"""This module contains the CausalEffect abstract class, as well as the concrete extension classes:
ExactValue, Positive, Negative, SomeEffect, NoEffect"""

from abc import ABC, abstractmethod

import numpy as np

from causal_testing.estimation.effect_estimate import EffectEstimate


class CausalEffect(ABC):
    """An abstract class representing an expected causal effect."""

    def __init__(self, effect_type: str = "direct"):
        self.effect_type = effect_type

    @abstractmethod
    def apply(self, effect_estimate: EffectEstimate) -> bool:
        """Abstract apply method that should return a bool representing if the result meets the outcome
        :param effect_estimate: EffectEstimate to be checked
        :return: Bool that is true if outcome is met
        """

    def __str__(self) -> str:
        return type(self).__name__

    def to_dict(self):
        """
        Convert the expected effect to a python dictionary for easy serialisation as JSON.

        :returns: A JSON serialisable dict representing the expected effect.
        """
        return {"name": self.__class__.__name__, "effect_type": self.effect_type}


class SomeEffect(CausalEffect):
    """An extension of CausalEffect representing that the expected causal effect should not be zero."""

    def apply(self, effect_estimate: EffectEstimate) -> bool:
        if effect_estimate.type in ("risk_ratio", "hazard_ratio", "unit_odds_ratio", "odds_ratio"):
            value_to_check = 1
        elif effect_estimate.type in ("coefficient", "ate"):
            value_to_check = 0
        else:
            raise ValueError(f"Test Value type {effect_estimate.type} is not valid for this CausalEffect")

        return (~((effect_estimate.ci_low <= value_to_check) & (value_to_check <= effect_estimate.ci_high))).all()


class NoEffect(CausalEffect):
    """
    An extension of CausalEffect representing that the expected causal effect should be zero.
    :param atol: Arithmetic tolerance. The test will pass if the absolute value of the causal effect is less than
    atol.
    :param ctol: Categorical tolerance. The test will pass if this proportion of categories pass.
    """

    def __init__(self, effect_type: str = "direct", atol: float = 0, ctol: float = 0.0):
        super().__init__(effect_type=effect_type)
        self.atol = atol
        self.ctol = ctol

    def apply(self, effect_estimate: EffectEstimate) -> bool:
        if effect_estimate.type in ("risk_ratio", "hazard_ratio", "unit_odds_ratio", "odds_ratio"):
            value_to_check = 1
        elif effect_estimate.type in ("coefficient", "ate"):
            value_to_check = 0
        else:
            raise ValueError(f"Test Value type {effect_estimate.type} is not valid for this CausalEffect")

        return sum(
            ((effect_estimate.ci_low <= value_to_check) & (value_to_check <= effect_estimate.ci_high))
            | (np.isclose(effect_estimate.value, value_to_check, atol=self.atol))
        ) / len(effect_estimate.value) >= (1 - self.ctol)

    def to_dict(self):
        """
        Convert the expected effect to a python dictionary for easy serialisation as JSON.

        :returns: A JSON serialisable dict representing the expected effect.
        """
        return super().to_dict() | {"atol": self.atol, "ctol": self.ctol}


class ExactValue(CausalEffect):
    """An extension of CausalEffect representing that the expected causal effect should be a specific value."""

    def __init__(
        self, value: float, effect_type: str = "direct", atol: float = None, ci_low: float = None, ci_high: float = None
    ):
        super().__init__(effect_type=effect_type)
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
                    f"Try specifying wider intervals or a value of atol smaller than the current vlaue {self.atol}."
                )

    def apply(self, effect_estimate: EffectEstimate) -> bool:
        close = np.isclose(effect_estimate.value, self.value, atol=self.atol)
        if effect_estimate.ci_valid and self.ci_low is not None and self.ci_high is not None:
            return (
                close.all()
                and (self.ci_low <= effect_estimate.ci_low).all()
                and (self.ci_high >= effect_estimate.ci_high).all()
            )
        return close.all()

    def __str__(self):
        return f"ExactValue: {self.value}±{self.atol}"

    def to_dict(self):
        """
        Convert the expected effect to a python dictionary for easy serialisation as JSON or CSV.

        :returns: A JSON serialisable dict representing the expected effect.
        """
        effect = {"value": self.value}
        if self.ci_low:
            effect["ci_low"] = self.ci_low
        if self.ci_low:
            effect["ci_high"] = self.ci_high
        if self.atol:
            effect["atol"] = self.atol
        return super().to_dict() | effect


class Positive(SomeEffect):
    """An extension of CausalEffect representing that the expected causal effect should be positive.
    Currently only single values are supported for the test value"""

    def apply(self, effect_estimate: EffectEstimate) -> bool:
        if len(effect_estimate.value) > 1:
            raise ValueError("Positive Effects are currently only supported on single float datatypes")
        if effect_estimate.type in {"ate", "coefficient"}:
            return any(0 < ci_low < ci_high for ci_low, ci_high in zip(effect_estimate.ci_low, effect_estimate.ci_high))
        if effect_estimate.type in ["risk_ratio", "unit_odds_ratio"]:
            return any(1 < ci_low < ci_high for ci_low, ci_high in zip(effect_estimate.ci_low, effect_estimate.ci_high))
        raise ValueError(f"Test Value type {effect_estimate.type} is not valid for this CausalEffect")


class Negative(SomeEffect):
    """An extension of CausalEffect representing that the expected causal effect should be negative.
    Currently only single values are supported for the test value"""

    def apply(self, effect_estimate: EffectEstimate) -> bool:
        if len(effect_estimate.value) > 1:
            raise ValueError("Negative Effects are currently only supported on single float datatypes")
        if effect_estimate.type in {"ate", "coefficient"}:
            return any(ci_low < ci_high < 0 for ci_low, ci_high in zip(effect_estimate.ci_low, effect_estimate.ci_high))
        if effect_estimate.type in ["risk_ratio", "unit_odds_ratio"]:
            return any(ci_low < ci_high < 1 for ci_low, ci_high in zip(effect_estimate.ci_low, effect_estimate.ci_high))
        # Dead code but necessary for pylint
        raise ValueError(f"Test Value type {effect_estimate.type} is not valid for this CausalEffect")
