import pandas as pd
from causal_testing.data_collection.data_collector import DataCollector
from causal_testing.specification.causal_specification import Scenario
from causal_testing.specification.constraint import UniformDistribution, AbsoluteValue


class ObservationalDataCollector(DataCollector):

    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario

    def collect_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.filter_valid_data()

    def filter_valid_data(self):
        variables = set(self.scenario.keys())

        # Check positivity
        if not variables.issubset(self.df.columns):
            missing_variables = variables - set(self.df.columns)
            raise IndexError(f'Positivity violation: missing data for variables {missing_variables}.')

        # For each variable in the scenario, remove rows where a constraint is violated
        for variable, constraint in self.scenario.items():
            self.df = self.df[self.df[variable].map(lambda x: value_meets_constraint(x, constraint))]


def value_meets_constraint(value, constraint):
    """
    Check that value-specific constraints are met.
    :param value: The value to check.
    :param constraint: The constraint to check the value against.
    :return: True or False depending on whether the value satisfies the constraint.
    """
    if isinstance(constraint, AbsoluteValue):
        return value == constraint.value
    elif isinstance(constraint, UniformDistribution):
        return constraint.min <= value <= constraint.max
