from z3 import ExprRef, substitute
from .variable import Variable, Input, Output, Meta
from tabulate import tabulate
from causal_testing.testing.causal_test_case import (
    CausalTestCase,
    AbstractCausalTestCase,
)


class Scenario:
    variables: {str: Variable}
    constraints: {ExprRef}
    test_cases: [CausalTestCase]
    abstract_test_cases: [CausalTestCase]

    def __init__(
        self,
        variables: {Variable},
        constraints: {ExprRef} = set(),
        test_cases: [CausalTestCase] = [],
        abstract_test_cases: [AbstractCausalTestCase] = [],
    ):
        self.variables = {v.name: v for v in variables}
        self.constraints = constraints
        self.test_cases = test_cases
        self.abstract_test_cases = abstract_test_cases

    def __repr__(self):
        """Returns a printable representational string of a scenario, e.g.
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

        repr = "Modelling scenario with variables:\n"
        repr += indent(
            tabulate(
                sorted(
                    [
                        (v.typestring(), v.name, v.datatype.__name__)
                        for v in self.variables.values()
                    ],
                )
            )
        )
        if len(self.constraints) > 0:
            repr += "\nAnd constraints:\n    "
            repr += "\n    ".join([str(c) for c in self.constraints])
        return repr

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
        self.treatment_variables = {
            k: self._fresh(v) for k, v in self.variables.items()
        }
        substitutions = {
            (self.variables[n].z3, self.treatment_variables[n].z3)
            for n in self.variables
        }
        treatment_constraints = {
            substitute(c, *substitutions) for c in self.constraints
        }
        self.constraints = self.constraints.union(treatment_constraints)

    def variables_of_type(self, t: type) -> {Variable}:
        """Get the set of scenario variables of a particular type, e.g. Inputs.

        :param type t: The type of variable to return, where t extends Variable.
        :return: A set of scenario variables of the supplied type.
        :rtype: {Variable}
        """
        return {v for v in self.variables.values() if isinstance(v, t)}

    def inputs(self) -> {Input}:
        """Get the set of scenario inputs.

        :return: The scenario inputs.
        :rtype: {Input}
        """
        return self.variables_of_type(Input)

    def outputs(self) -> {Output}:
        """Get the set of scenario outputs.

        :return: The scenario outputs.
        :rtype: {Output}
        """
        return self.variables_of_type(Output)

    def metas(self) -> {Meta}:
        """Get the set of scenario metavariables.

        :return: The scenario metavariables.
        :rtype: {Input}
        """
        return self.variables_of_type(Meta)

    def add_variable(v: Variable) -> None:
        self.variables[v.name]: v
