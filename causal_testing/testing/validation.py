"""This module contains the CausalValidator class for performing Quantitive Bias Analysis techniques"""
import math
import numpy as np
from scipy.stats import t
from statsmodels.regression.linear_model import RegressionResultsWrapper


class CausalValidator:
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

    def estimate_e_value(
        self, risk_ratio, confidence_intervals: tuple[float, float]
    ) -> tuple[float, tuple[float, float]]:
        """Calculate the E value from a risk ratio. This allow
        the user to identify how large a risk an unidentified confounding
        variable would need to be to nullify the causal relationship
        under test."""

        if risk_ratio >= 1:
            e = risk_ratio + math.sqrt(risk_ratio * (risk_ratio - 1))

            lower_limit = confidence_intervals[0]
            if lower_limit <= 1:
                lower_limit = 1
            else:
                lower_limit = lower_limit + math.sqrt(lower_limit * (lower_limit - 1))

            return (e, (lower_limit, 1))

        else:
            risk_ratio_prime = 1 / risk_ratio
            e = risk_ratio_prime + math.sqrt(risk_ratio_prime * (risk_ratio_prime - 1))

            upper_limit = confidence_intervals[1]
            if upper_limit >= 1:
                upper_limit = 1
            else:
                upper_limit_prime = 1 / upper_limit
                upper_limit = upper_limit_prime + math.sqrt(upper_limit_prime * (upper_limit_prime - 1))

            return (e, (1, upper_limit))
