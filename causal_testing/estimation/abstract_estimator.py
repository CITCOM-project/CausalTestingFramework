"""This module contains the Estimator abstract class"""

import logging
from abc import ABC, abstractmethod
from typing import Any

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

    2) estimate_*: The Causal Testing Framework expects causal effects to be calculated by methods that start with
    `estimate_`, followed by the name of the causal effect measure being estimated, for example `ate` or `risk_ratio`.
    Naming methods this way enables estimators to hook nicely into the endpoints further up the chain.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        # pylint: disable=R0801
        self,
        treatment_variable: str,
        outcome_variable: str,
        control_value: float = None,
        treatment_value: float = None,
        adjustment_set: set = None,
        adjustment_config: dict[str, Any] = None,
        alpha: float = 0.05,
    ):

        self.treatment_variable = treatment_variable
        self.outcome_variable = outcome_variable
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.alpha = alpha
        self.adjustment_config = {} if adjustment_config is None else adjustment_config
        self.adjustment_set = (
            set(self.adjustment_config) if adjustment_set is None else adjustment_set.union(set(self.adjustment_config))
        )
        self.modelling_assumptions = []
        self.add_modelling_assumptions()

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """

    def to_dict(self) -> dict:
        """
        Convert the estimator to a python dictionary for easy serialisation as JSON or CSV.

        :returns: A JSON serialisable dict representing the estimator.
        """
        result = {
            "name": self.__class__.__name__,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "alpha": self.alpha,
            "adjustment_set": sorted(self.adjustment_set),
        }
        if self.adjustment_config:
            result["adjustment_config"] = self.adjustment_config
        if self.control_value is not None:
            result["control_value"] = self.control_value
        if self.treatment_value is not None:
            result["treatment_value"] = self.treatment_value
        return result
