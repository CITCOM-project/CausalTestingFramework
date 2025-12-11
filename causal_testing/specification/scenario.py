"""This module holds the Scenario Class"""

from collections.abc import Iterable
from dataclasses import dataclass


from .variable import Variable


@dataclass
class Scenario:
    """A scenario defines the setting by listing the endogenous variables, their
    datatypes, distributions, and any constraints over them. This is a common
    practice in CI and is analogous to an investigator specifying “we are
    interested in individuals over 40 who regularly eat cheese” or whatever. A
    scenario, here, is not a specific test case; it just defines the population
    of interest, in our case “runs of the model with parameters meeting the
    constraints”. The model may have other inputs/outputs which the investigator
    may choose to leave out. These are then exogenous variables and behave
    accordingly.

    :param {Variable} variables: The set of endogenous variables.
    :param {str} constraints: The set of constraints relating the endogenous variables.
    :attr variables:
    :attr constraints:
    """

    def __init__(self, variables: Iterable[Variable], constraints: set[str] = None):
        self.variables = {v.name: v for v in variables}
        if constraints is not None:
            self.constraints = set(constraints)
        else:
            self.constraints = set()

    def hidden_variables(self) -> set[Variable]:
        """Get the set of hidden variables

        :return The variables marked as hidden.
        :rtype: {Variable}
        """
        return {v for v in self.variables.values() if v.hidden}
