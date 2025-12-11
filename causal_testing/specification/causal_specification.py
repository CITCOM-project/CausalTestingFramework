"""This module holds the CausalSpecification data class."""

from dataclasses import dataclass
from collections.abc import Iterable

from causal_testing.specification.causal_dag import CausalDAG
from .variable import Variable


@dataclass
class CausalSpecification:
    """
    Data class storing the Causal Specification, made up of the modelling scenario and causal DAG).
    A scenario defines the setting by listing the endogenous variables, their
    datatypes, distributions, and any constraints over them. This is a common
    practice in CI and is analogous to an investigator specifying "we are
    interested in individuals over 40 who regularly eat cheese" or whatever. A
    scenario, here, is not a specific test case; it just defines the population
    of interest, in our case "runs of the model with parameters meeting the
    constraints". The model may have other inputs/outputs which the investigator
    may choose to leave out. These are then exogenous variables and behave accordingly.

    :param {Variable} variables: The set of endogenous variables.
    :param {str} causal_dag: The causal DAG.
    :param {str} constraints: The set of constraints relating the endogenous variables.
    """

    def __init__(self, variables: Iterable[Variable], causal_dag: CausalDAG, constraints: set[str] = None):
        self.variables = {v.name: v for v in variables}
        self.causal_dag = causal_dag
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

    def __str__(self):
        return f"Scenario: {self.scenario}\nCausal DAG:\n{self.causal_dag}"
