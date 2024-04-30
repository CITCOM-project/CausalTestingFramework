"""This module holds the Scenario Class"""

from collections.abc import Iterable, Mapping

from tabulate import tabulate
from z3 import ExprRef, substitute

from .variable import Input, Meta, Output, Variable


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
    :param {ExprRef} constraints: The set of constraints relating the endogenous variables.
    :attr variables:
    :attr constraints:
    """

    variables: Mapping[str, Variable]
    constraints: set[ExprRef]

    def __init__(self, variables: Iterable[Variable] = None, constraints: set[ExprRef] = None):
        if variables is not None:
            self.variables = {v.name: v for v in variables}
        else:
            self.variables = {}
        if constraints is not None:
            self.constraints = set(constraints)
        else:
            self.constraints = set()

        self.prime = {}
        self.unprime = {}
        self.treatment_variables = {}

    def __str__(self):
        """Returns a printable string of a scenario, e.g.
        Modelling scenario with variables:
            ------  ---------------  -----
            INPUT   location         str
            INPUT   n_days           int
            INPUT   pop_size         int
            META    average_age      int
            META    household_size   float
            OUTPUT  cum_deaths       int
            OUTPUT  cum_infections   int
            OUTPUT  cum_quarantined  int
            ------  ---------------  -----
        And constraints:
            n_days <= 365
            n_days >= 60
            average_age > 0

        :return: A printable version of a scenario.
        :rtype: str
        """

        def indent(txt, spaces=4):
            return "\n".join(" " * spaces + ln for ln in txt.splitlines())

        string = "Modelling scenario with variables:\n"
        string += indent(
            tabulate(
                sorted(
                    [(v.typestring(), v.name, v.datatype.__name__) for v in self.variables.values()],
                )
            )
        )
        if len(self.constraints) > 0:
            string += "\nAnd constraints:\n    "
            string += "\n    ".join([str(c) for c in self.constraints])
        return string

    def _fresh(self, variable: Variable) -> Variable:
        """Create a "primed" version of the given variable to represent the CI
        treatment values, e.g. if a variable v represents the control value,
        then a variable v' will be created to represent the treatment value.

        :param Variable variable: The variable to "prime".
        :return: A fresh "primed" variable.
        :rtype: Variable
        """
        vname = variable.name
        while vname in self.variables:
            vname += "'"
        return variable.copy(vname)

    def setup_treatment_variables(self) -> None:
        """Create a mirror of the current variable set with "primed" variables
        to represent the treatment values. Corresponding constraints are added
        to the contraint set such that the "primed" variables are constrained in
        the same way as their unprimed counterparts.
        """
        for k, v in self.variables.items():
            v_prime = self._fresh(v)
            self.treatment_variables[k] = v_prime
            self.prime[k] = v_prime.name
            self.unprime[v_prime.name] = k

        substitutions = {(self.variables[n].z3, self.treatment_variables[n].z3) for n in self.variables}
        treatment_constraints = {substitute(c, *substitutions) for c in self.constraints}
        self.constraints = self.constraints.union(treatment_constraints)

    def variables_of_type(self, t: type) -> set[Variable]:
        """Get the set of scenario variables of a particular type, e.g. Inputs.

        :param type t: The type of variable to return, where t extends Variable.
        :return: A set of scenario variables of the supplied type.
        :rtype: {Variable}
        """
        return {v for v in self.variables.values() if isinstance(v, t)}

    def inputs(self) -> set[Input]:
        """Get the set of scenario inputs.

        :return: The scenario inputs.
        :rtype: {Input}
        """
        return self.variables_of_type(Input)

    def outputs(self) -> set[Output]:
        """Get the set of scenario outputs.

        :return: The scenario outputs.
        :rtype: {Output}
        """
        return self.variables_of_type(Output)

    def metas(self) -> set[Meta]:
        """Get the set of scenario metavariables.

        :return: The scenario metavariables.
        :rtype: {Input}
        """
        return self.variables_of_type(Meta)

    def hidden_variables(self) -> set[Variable]:
        """Get the set of hidden variables

        :return The variables marked as hidden.
        :rtype: {Variable}
        """
        return {v for v in self.variables.values() if v.hidden}

    def add_variable(self, v: Variable) -> None:
        """Add variable to variables attribute
        :param v: Variable to be added
        """
        self.variables[v.name]: v
