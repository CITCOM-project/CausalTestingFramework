import logging
from typing import Any

from causal_testing.specification.variable import Variable
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.testing.base_causal_test import BaseCausalTest

logger = logging.getLogger(__name__)


class CausalTestCase(BaseCausalTest):
    """)
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y.
    """

    def __init__(
        self,
        base_causal_test: BaseCausalTest,
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
        self.control_input_configuration = {base_causal_test.treatment_variable: control_value}
        self.expected_causal_effect = expected_causal_effect
        self.outcome_variable = base_causal_test.outcome_variable
        self.treatment_input_configuration = {base_causal_test.treatment_variable: treatment_value}
        self.estimate_type = estimate_type
        self.effect = base_causal_test.effect
        if effect_modifier_configuration:
            self.effect_modifier_configuration = effect_modifier_configuration
        else:
            self.effect_modifier_configuration = dict()
        assert (
            self.control_input_configuration.keys() == self.treatment_input_configuration.keys()
        ), "Control and treatment input configurations must have the same keys."

    def get_treatment_variables(self):
        """Return a list of the treatment variables (as strings) for this causal test case."""
        return [v.name for v in self.control_input_configuration]

    def get_outcome_variables(self):
        """Return a list of the outcome variables (as strings) for this causal test case."""
        return [v.name for v in self.outcome_variables]

    def get_control_values(self):
        """Return a list of the control values for each treatment variable in this causal test case."""
        return list(self.control_input_configuration.values())

    def get_treatment_values(self):
        """Return a list of the treatment values for each treatment variable in this causal test case."""
        return list(self.treatment_input_configuration.values())

    def __str__(self):
        treatment_config = {k.name: v for k, v in self.treatment_input_configuration.items()}
        control_config = {k.name: v for k, v in self.control_input_configuration.items()}
        return (
            f"Running {treatment_config} instead of {control_config} should cause the following "
            f"changes to {self.outcome_variables}: {self.expected_causal_effect}."
        )
