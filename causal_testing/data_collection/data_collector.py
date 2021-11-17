from abc import ABC, abstractmethod
from causal_testing.specification.causal_specification import Scenario
from causal_testing.specification.constraint import UniformDistribution, AbsoluteValue
import pandas as pd


class DataCollector(ABC):

    @abstractmethod
    def collect_data(self, **kwargs):
        """
        Populate the dataframe with execution data.
        :return:
        """
        ...


class ExperimentalDataCollector(DataCollector):
    """
    Users should implement these methods to collect data from their system directly.
    """
    def __init__(self, control_input_configuration, treatment_input_configuration, n_repeats=1):
        super().__init__()
        self.control_input_configuration = control_input_configuration
        self.treatment_input_configuration = treatment_input_configuration
        self.n_repeats = n_repeats

    @abstractmethod
    def collect_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def run_system_with_input_configuration(self, input_configuration):
        pass


class ObservationalDataCollector(DataCollector):

    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario

    def collect_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df = self.filter_valid_data(df)
        return df

    def filter_valid_data(self, df):
        """
        Check if observational df is valid for a given scenario.
        """
        variables = set(self.scenario.keys())

        # Check positivity
        if not variables.issubset(df.columns):
            missing_variables = variables - set(df.columns)
            raise IndexError(f'Positivity violation: missing data for variables {missing_variables}.')

        # For each variable in the scenario, remove rows where a constraint is violated
        for variable, constraint in self.scenario.items():
            df = df[df[variable].map(lambda x: value_meets_constraint(x, constraint))]

        return df


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
