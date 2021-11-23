from causal_testing.specification.constraint import Constraint


class Scenario(dict):
    """
    Given a system with X distinct inputs, a scenario is a series of constraints placed over a subset of these
    inputs that characterises some use-case of the system-under-test.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def add_constraint(self, input_variable: str, constraint: Constraint):
        self[input_variable] = constraint

    def add_constraints(self, constraints_dict: dict):
        self.update(constraints_dict)
