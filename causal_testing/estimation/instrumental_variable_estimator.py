"""This module contains the InstrumentalVariableEstimator class, for estimating
continuous outcomes with unobservable confounding."""

import logging
from math import ceil
import pandas as pd
import statsmodels.api as sm

from causal_testing.estimation.abstract_estimator import Estimator

logger = logging.getLogger(__name__)


class InstrumentalVariableEstimator(Estimator):
    """
    Carry out estimation using instrumental variable adjustment rather than conventional adjustment. This means we do
    not need to observe all confounders in order to adjust for them. A key assumption here is linearity.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        # pylint: disable=duplicate-code
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        instrument: str,
        df: pd.DataFrame = None,
        alpha: float = 0.05,
        query: str = "",
    ):
        super().__init__(
            treatment=treatment,
            treatment_value=treatment_value,
            control_value=control_value,
            adjustment_set=adjustment_set,
            outcome=outcome,
            df=df,
            effect_modifiers=None,
            alpha=alpha,
            query=query,
        )

        self.instrument = instrument

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions.append(
            """The instrument and the treatment, and the treatment and the outcome must be
        related linearly in the form Y = aX + b."""
        )
        self.modelling_assumptions.append(
            """The three IV conditions must hold
            (i) Instrument is associated with treatment
            (ii) Instrument does not affect outcome except through its potential effect on treatment
            (iii) Instrument and outcome do not share causes
        """
        )

    def estimate_iv_coefficient(self, df) -> float:
        """
        Estimate the linear regression coefficient of the treatment on the
        outcome.
        """
        # Estimate the total effect of instrument I on outcome Y = abI + c1
        ab = sm.OLS(df[self.outcome], df[[self.instrument]]).fit().params[self.instrument]

        # Estimate the direct effect of instrument I on treatment X = aI + c1
        a = sm.OLS(df[self.treatment], df[[self.instrument]]).fit().params[self.instrument]

        # Estimate the coefficient of I on X by cancelling
        return ab / a

    def estimate_coefficient(self, bootstrap_size=100) -> tuple[pd.Series, list[pd.Series, pd.Series]]:
        """
        Estimate the unit ate (i.e. coefficient) of the treatment on the
        outcome.
        """
        bootstraps = sorted(
            [self.estimate_iv_coefficient(self.df.sample(len(self.df), replace=True)) for _ in range(bootstrap_size)]
        )
        bound = ceil((bootstrap_size * self.alpha) / 2)
        ci_low = pd.Series(bootstraps[bound])
        ci_high = pd.Series(bootstraps[bootstrap_size - bound])

        return pd.Series(self.estimate_iv_coefficient(self.df)), [ci_low, ci_high]
