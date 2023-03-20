"""This module contains the CausalValidator class for performing Quantitive Bias Analysis techniques"""
import math
import numpy as np
from scipy.stats import t
from statsmodels.regression.linear_model import RegressionResultsWrapper


class CausalValidator:
    """A suite of validation tools to perform Quantitive Bias Analysis to back up causal claims"""

    def estimate_robustness(self, model: RegressionResultsWrapper, q=1, alpha=1):
        """Calculate the robustness of a linear regression model. This allow
        the user to identify how large an unidentified confounding variable
        would need to be to nullify the causal relationship under test."""

        dof = model.df_resid
        t_values = model.tvalues

        fq = q * abs(t_values / math.sqrt(dof))
        f_crit = abs(t.ppf(alpha / 2, dof - 1)) / math.sqrt(dof - 1)
        fqa = fq - f_crit

        rv = 0.5 * (np.sqrt(fqa**4 + (4 * fqa**2)) - fqa**2)

        return rv

    def estimate_e_value(self, risk_ratio: float) -> float:
        """Calculate the E value from a risk ratio. This allow
        the user to identify how large a risk an unidentified confounding
        variable would need to be to nullify the causal relationship
        under test."""

        if risk_ratio >= 1:
            return risk_ratio + math.sqrt(risk_ratio * (risk_ratio - 1))

        risk_ratio_prime = 1 / risk_ratio
        return risk_ratio_prime + math.sqrt(risk_ratio_prime * (risk_ratio_prime - 1))

    def estimate_e_value_using_ci(self, risk_ratio: float, confidence_intervals: tuple[float, float]) -> float:
        """Calculate the E value from a risk ratio and it's confidence intervals.
        This allow the user to identify how large a risk an unidentified
        confounding variable would need to be to nullify the causal relationship
        under test."""

        if risk_ratio >= 1:
            lower_limit = confidence_intervals[0]
            e = 1
            if lower_limit > 1:
                e = lower_limit + math.sqrt(lower_limit * (lower_limit - 1))

            return e

        upper_limit = confidence_intervals[1]
        e = 1
        if upper_limit < 1:
            upper_limit_prime = 1 / upper_limit
            e = upper_limit_prime + math.sqrt(upper_limit_prime * (upper_limit_prime - 1))

        return e
