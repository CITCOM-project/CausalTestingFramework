"""
This module contains the EffectEstimate dataclass.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class EffectEstimate:
    """
    A dataclass to hold the value and confidence intervals of a causal effect estimate

    :ivar type: The type of estimate, e.g. ate, or risk_ratio
                (used to determine whether the estimate matches the expected effect)
    :ivar value: The estimated causal effect
    :ivar ci_low: The lower confidence interval
    :ivar ci_high: The upper confidence interval
    """

    def __init__(
        self, effect_measure: str, effect_estimate: pd.Series, ci_low: pd.Series = None, ci_high: pd.Series = None
    ):
        self.effect_measure = effect_measure
        self.effect_estimate = pd.Series(effect_estimate)
        self.ci_low = pd.Series(ci_low) if ci_low is not None else None
        self.ci_high = pd.Series(ci_high) if ci_high is not None else None

    def ci_valid(self) -> bool:
        """Return whether or not the result has valid confidence invervals"""
        return (
            self.ci_low is not None
            and self.ci_high is not None
            and not (pd.isnull(self.ci_low).any() or pd.isnull(self.ci_high).any())
        )

    def to_dict(self) -> dict:
        """Return representation as a dict."""
        d = {
            "effect_measure": self.effect_measure,
            "effect_estimate": self.effect_estimate.to_dict(),
        }
        if self.ci_valid():
            return d | {"ci_low": self.ci_low.to_dict(), "ci_high": self.ci_high.to_dict()}
        return d

    def to_df(self) -> pd.DataFrame:
        """Return representation as a pandas dataframe."""
        return pd.DataFrame({"effect_estimate": self.effect_estimate, "ci_low": self.ci_low, "ci_high": self.ci_high})
