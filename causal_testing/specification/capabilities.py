"""
This module contains the Capability and TreatmentSequence classes to implement
treatment sequences that operate over time.
"""

from typing import Any
from causal_testing.specification.variable import Variable


class Capability:
    """
    Data class to encapsulate temporal interventions.
    """

    def __init__(self, variable: Variable, value: Any, start_time: int, end_time: int):
        self.variable = variable
        self.value = value
        self.start_time = start_time
        self.end_time = end_time

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.variable == other.variable
            and self.value == other.value
            and self.start_time == other.start_time
            and self.end_time == other.end_time
        )

    def __repr__(self):
        return f"({self.variable}, {self.value}, {self.start_time}-{self.end_time})"


class TreatmentSequence:
    """
    Class to represent a list of capabilities, i.e. a treatment regime.
    """

    def __init__(self, timesteps_per_intervention, capabilities):
        self.timesteps_per_intervention = timesteps_per_intervention
        self.capabilities = [
            Capability(var, val, t, t + timesteps_per_intervention)
            for (var, val), t in zip(
                capabilities,
                range(
                    timesteps_per_intervention,
                    (len(capabilities) * timesteps_per_intervention) + 1,
                    timesteps_per_intervention,
                ),
            )
        ]
        # This is a bodge so that causal test adequacy works
        self.name = tuple(c.variable for c in self.capabilities)

    def set_value(self, index: int, value: float):
        """
        Set the value of capability at the given index.
        :param index - the index of the element to update.
        :param value - the desired value of the capability.
        """
        self.capabilities[index].value = value

    def copy(self):
        """
        Return a deep copy of the capability list.
        """
        strategy = TreatmentSequence(
            self.timesteps_per_intervention,
            [(c.variable, c.value) for c in self.capabilities],
        )
        return strategy

    def total_time(self):
        """
        Calculate the total duration of the treatment strategy.
        """
        return (len(self.capabilities) + 1) * self.timesteps_per_intervention
