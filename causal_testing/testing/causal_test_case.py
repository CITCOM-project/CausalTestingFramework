from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_outcome import CausalTestOutcome
from causal_testing.specification.variable import Variable

import logging
logger = logging.getLogger(__name__)


class CausalTestCase:
    """
    A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is an intervention, and
    Y is the expected causal effect on a particular output. The goal of a causal test case is to test whether the
    intervention Delta made to the input configuration X causes the model-under-test to produce the expected change
    in Y.
    """

    def __init__(self, control_input_configuration: {Variable: any}, expected_causal_effect: CausalTestOutcome,
                outcome_variables: {Variable}, intervention: Intervention = None,
                treatment_input_configuration: {Variable: any} = None):
        """
        When a CausalTestCase is initialised, it takes the intervention and applies it to the input configuration to
        create two distinct input configurations: a control input configuration and a treatment input configuration.
        The former is the input configuration before applying the intervention and the latter is the input configuration
        after applying the intervention.

        :param {Variable: any} control_input_configuration: The input configuration representing the control values of
        the treatment variables.
        :param CausalTestOutcome The expected outcome.
        :param Intervention intervention: The metamorphic operator which transforms the control configuration to the
        treatment configuration. Defaults to None.
        :param {Variable: any} treatment_input_configuration: The input configuration representing the treatment
        values of the treatment variables.
        """
        assert (
            intervention is None or treatment_input_configuration is None
        ), "Cannot define both treatment configuration and intervention."
        assert (
            intervention is not None or treatment_input_configuration is not None
        ), "Must define either a treatment configuration or intervention."
        if intervention is not None:
            assert isinstance(intervention, Intervention), f"Invervention must be an instance of class Intervention not {type(intervention)}"

        self.control_input_configuration = control_input_configuration
        self.expected_causal_effect = expected_causal_effect
        self.intervention = intervention
        self.outcome_variables = outcome_variables
        self.treatment_input_configuration = treatment_input_configuration
        if intervention:
            self.treatment_input_configuration = intervention.apply(
                self.control_input_configuration
            )

    def __str__(self):
        if self.intervention is not None:
            return (f"Applying {self.intervention} to {self.control_input_configuration} should cause the following "
                    f"changes to {self.outcome_variables}: {self.expected_causal_effect}.")
        else:
            treatment_config = {k.name: v for k, v in self.treatment_input_configuration.items()}
            control_config = {k.name: v for k, v in self.control_input_configuration.items()}
            return (f"Running {treatment_config} instead of {control_config} should cause the following "
                    f"changes to {self.outcome_variables}: {self.expected_causal_effect}.")
