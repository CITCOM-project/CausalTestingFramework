"""This module contains the CausalTestResult class, which is a container for the results of a causal test."""

import pandas as pd

from causal_testing.estimation.abstract_estimator import Estimator
from causal_testing.estimation.effect_estimate import EffectEstimate


class CausalTestResult:
    """A container to hold the results of a causal test case. Every causal test case provides a point estimate of
    the ATE, given a particular treatment, outcome, and adjustment set. Some but not all estimators can provide
    confidence intervals."""

    def __init__(
        self,
        estimator: Estimator,
        effect_estimate: EffectEstimate,
        adequacy=None,
        error_message: str = None,
    ):
        self.estimator = estimator
        self.adequacy = adequacy
        if estimator.adjustment_set:
            self.adjustment_set = estimator.adjustment_set
        else:
            self.adjustment_set = set()
        self.effect_estimate = effect_estimate
        self.error_message = error_message

    def __str__(self):
        result_str = str(self.effect_estimate.value.to_dict())
        treatment = self.estimator.base_test_case.treatment_variable.name
        base_str = (
            f"Causal Test Result\n==============\n"
            f"Treatment: {treatment}\n"
            f"Control value: {self.estimator.control_value}\n"
            f"Treatment value: {self.estimator.treatment_value}\n"
            f"Outcome: {self.estimator.base_test_case.outcome_variable.name}\n"
            f"Adjustment set: {self.adjustment_set}\n"
        )
        if hasattr(self.estimator, "formula"):
            base_str += f"Formula: {self.estimator.formula}\n"
        base_str += f"{self.effect_estimate.type}: {result_str}\n"
        confidence_str = ""
        if self.effect_estimate.ci_valid():
            ci_str = f"CI low: {self.effect_estimate.ci_low.to_dict}\n"
            ci_str += f"CI high: {self.effect_estimate.ci_high.to_dict}\n"
            confidence_str += f"Confidence intervals:{ci_str}\n"
            confidence_str += f"Alpha:{self.estimator.alpha}\n"
        adequacy_str = ""
        if self.adequacy:
            adequacy_str = str(self.adequacy)
        return base_str + confidence_str + adequacy_str

    def to_dict(self, json=False):
        """Return result contents as a dictionary
        :return: Dictionary containing contents of causal_test_result
        """
        base_dict = {
            "treatment": (
                self.estimator.base_test_case.treatment_variable.name
                if self.estimator.base_test_case.treatment_variable is not None
                else None
            ),
            "control_value": self.estimator.control_value,
            "treatment_value": self.estimator.treatment_value,
            "outcome": self.estimator.base_test_case.outcome_variable.name,
            "adjustment_set": list(self.adjustment_set) if json else self.adjustment_set,
            "effect_measure": self.effect_estimate.type,
        } | self.effect_estimate.to_dict()
        if self.adequacy:
            base_dict["adequacy"] = self.adequacy.to_dict()
        return base_dict
