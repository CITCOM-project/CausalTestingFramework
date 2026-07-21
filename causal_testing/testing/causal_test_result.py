"""This module contains the CausalTestResult class, which is a container for the results of a causal test."""

from dataclasses import dataclass
from enum import Enum

from causal_testing.estimation.effect_estimate import EffectEstimate

TestOutcome = Enum("TestOutcome", [("PASS", 2), ("FAIL", 0), ("INESTIMABLE", 1)])


@dataclass
class CausalTestResult:
    """A container to hold the results of a causal test case. Every causal test case provides a point estimate of
    the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
    confidence intervals."""

    def __init__(
        self,
        effect_estimate: EffectEstimate,
        outcome: TestOutcome,
        adequacy=None,
        error_message: str = None,
    ):
        self.effect_estimate = effect_estimate
        self.outcome = outcome
        self.adequacy = adequacy
        self.error_message = error_message

    def to_dict(self):
        """
        Convert the result to a python dictionary for easy serialisation as JSON.

        :returns: A JSON serialisable dict representing the test result.
        """

        outcome = {"outcome": self.outcome.name, "passed": self.outcome == TestOutcome.PASS}
        if self.error_message:
            outcome["error_message"] = self.error_message

        effect_estimate = self.effect_estimate.to_dict()

        adequacy = self.adequacy.to_dict() if self.adequacy else {}

        return outcome | effect_estimate | {"adequacy": adequacy}
