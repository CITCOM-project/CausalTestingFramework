"""This module contains the Estimator abstract class"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from causal_testing.testing.base_test_case import BaseTestCase

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
        base_test_case: BaseTestCase,
        control_value: float = None,
        treatment_value: float = None,
        adjustment_set: set = None,
        effect_modifiers: dict[str, Any] = None,
        alpha: float = 0.05,
    ):
        self.base_test_case = base_test_case
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.adjustment_set = adjustment_set
        self.alpha = alpha

        if effect_modifiers is None:
            self.effect_modifiers = {}
        else:
            self.effect_modifiers = effect_modifiers
        self.modelling_assumptions = []
        self.add_modelling_assumptions()
        logger.debug("Effect Modifiers: %s", self.effect_modifiers)

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
        result = {"name": self.__class__.__name__, "alpha": self.alpha, "adjustment_set": sorted(self.adjustment_set)}
        if self.effect_modifiers:
            result["effect_modifiers"] = self.effect_modifiers
        if self.control_value is not None:
            result["control_value"] = self.control_value
        if self.treatment_value is not None:
            result["treatment_value"] = self.treatment_value
        return result
