"""
This module contains the EffectEstimate dataclass.
"""

import pandas as pd
from dataclasses import dataclass


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

    type: str
    value: pd.Series
    ci_low: pd.Series = None
    ci_high: pd.Series = None

    def ci_valid(self) -> bool:
        """Return whether or not the result has valid confidence invervals"""
        return (
            self.ci_low is not None
            and self.ci_high is not None
            and not (pd.isnull(self.ci_low).any() or pd.isnull(self.ci_high).any())
        )

    def to_dict(self) -> dict:
        """Return representation as a dict."""
        d = {"effect_measure": self.type, "effect_estimate": self.value.to_dict()}
        if self.ci_valid():
            return d | {"ci_low": self.ci_low.to_dict(), "ci_high": self.ci_high.to_dict()}
        return d

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({"effect_estimate": self.value, "ci_low": self.ci_low, "ci_high": self.ci_high})
