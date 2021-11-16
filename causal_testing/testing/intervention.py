from abc import ABC, abstractmethod


class Intervention(ABC):
    """
    An intervention is an object which manipulates the input configuration of the scenario-under-test. It must define
    a method which takes the input configuration, does something to it, and returns a modified input configuration.

    This provides a causal test case with two input configurations to compare: a control input configuration (the
    original) and a treatment input configuration (the modified). The causal test case then requires data for the
    execution of each of these input configurations to obtain the causal effect of this intervention.
    """

    def __init__(self, treatment_variables: tuple, treatment_values: tuple):
        self.treatment_variables = treatment_variables
        self.treatment_values = treatment_values

    @abstractmethod
    def apply(self, input_configuration: dict):
        """
        Take an input configuration and modify it in a particular way. It is the effect of this change which the
        causal test case will focus on.
        :param input_configuration: Input configuration for the scenario-under-test.
        :return treatment_input_configuration: a modified input configuration.
        """
        second_input_configuration = input_configuration.copy()
        treatment_input_configuration = second_input_configuration[self.treatment_variables] = self.treatment_values
        return treatment_input_configuration
