"""This module contains the Estimator abstract class"""

import logging
from abc import ABC, abstractmethod
from typing import Any
from math import ceil

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix  # pylint: disable = no-name-in-module
from patsy import ModelDesc
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from lifelines import CoxPHFitter

from causal_testing.specification.variable import Variable
from causal_testing.specification.capabilities import TreatmentSequence, Capability

logger = logging.getLogger(__name__)


class Estimator(ABC):
    # pylint: disable=too-many-instance-attributes
    """An estimator contains all of the information necessary to compute a causal estimate for the effect of changing
    a set of treatment variables to a set of values.

    All estimators must implement the following two methods:

    1) add_modelling_assumptions: The validity of a model-assisted causal inference result depends on whether
    the modelling assumptions imposed by a model actually hold. Therefore, for each model, is important to state
    the modelling assumption upon which the validity of the results depend. To achieve this, the estimator object
    maintains a list of modelling assumptions (as strings). If a user wishes to implement their own estimator, they
    must implement this method and add all assumptions to the list of modelling assumptions.

    2) estimate_ate: All estimators must be capable of returning the average treatment effect as a minimum. That is, the
    average effect of the intervention (changing treatment from control to treated value) on the outcome of interest
    adjusted for all confounders.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        treatment: str,
        treatment_value: float,
        control_value: float,
        adjustment_set: set,
        outcome: str,
        df: pd.DataFrame = None,
        effect_modifiers: dict[str:Any] = None,
        alpha: float = 0.05,
        query: str = "",
    ):
        self.treatment = treatment
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.adjustment_set = adjustment_set
        self.outcome = outcome
        self.alpha = alpha
        self.df = df.query(query) if query else df

        if effect_modifiers is None:
            self.effect_modifiers = {}
        elif isinstance(effect_modifiers, dict):
            self.effect_modifiers = effect_modifiers
        else:
            raise ValueError(f"Unsupported type for effect_modifiers {effect_modifiers}. Expected iterable")
        self.modelling_assumptions = []
        if query:
            self.modelling_assumptions.append(query)
        self.add_modelling_assumptions()
        logger.debug("Effect Modifiers: %s", self.effect_modifiers)

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """

    def compute_confidence_intervals(self) -> list[float, float]:
        """
        Estimate the 95% Wald confidence intervals for the effect of changing the treatment from control values to
        treatment values on the outcome.
        :return: 95% Wald confidence intervals.
        """
