import logging
from typing import Any

from causal_testing.specification.variable import Variable
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.testing.base_test_case import BaseTestCase

logger = logging.getLogger(__name__)


class CausalTestCase:
    """
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y.
    """

    def __init__(
        self,
        base_test_case: BaseTestCase,
        expected_causal_effect: CausalTestOutcome,
        control_value: Any,
        treatment_value: Any = None,
        estimate_type: str = "ate",
        effect_modifier_configuration: dict[Variable:Any] = None,
    ):
        """
        When a CausalTestCase is initialised, it takes the intervention and applies it to the input configuration to
        create two distinct input configurations: a control input configuration and a treatment input configuration.
        The former is the input configuration before applying the intervention and the latter is the input configuration
        after applying the intervention.

        :param control_input_configuration: The input configuration representing the control values of the treatment
        variables.
        :param treatment_input_configuration: The input configuration representing the treatment values of the treatment
        variables. That is, the input configuration *after* applying the intervention.
        """
        self.base_test_case = base_test_case
        self.control_value = control_value
        self.expected_causal_effect = expected_causal_effect
        self.outcome_variable = base_test_case.outcome_variable
        self.treatment_variable = base_test_case.treatment_variable
        self.treatment_value = treatment_value
        self.estimate_type = estimate_type
        self.effect = base_test_case.effect

        if effect_modifier_configuration:
            self.effect_modifier_configuration = effect_modifier_configuration
        else:
            self.effect_modifier_configuration = dict()

    def get_treatment_variable(self):
        """Return a list of the treatment variables (as strings) for this causal test case."""
        return self.treatment_variable.name

    def get_outcome_variable(self):
        """Return a list of the outcome variables (as strings) for this causal test case."""
        return self.outcome_variable.name

    def get_control_value(self):
        """Return a list of the control values for each treatment variable in this causal test case."""
        return self.control_value

    def get_treatment_value(self):
        """Return a list of the treatment values for each treatment variable in this causal test case."""
        return self.treatment_value

    def __str__(self):
        treatment_config = {self.treatment_variable.name: self.treatment_value}
        control_config = {self.treatment_variable.name: self.control_value}
        outcome_variable = {self.outcome_variable}
        return (
            f"Running {treatment_config} instead of {control_config} should cause the following "
            f"changes to {outcome_variable}: {self.expected_causal_effect}."
        )
