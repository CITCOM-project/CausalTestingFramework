from abc import ABC, abstractmethod
from causal_testing.specification.causal_specification import Scenario

from typing import Union
import pandas as pd
from z3 import ExprRef


class DataCollector(ABC):
    @abstractmethod
    def collect_data(self, **kwargs) -> pd.DataFrame:
        """
        Populate the dataframe with execution data.
        :return df: A pandas dataframe containing execution data for the system-under-test.
        """
        pass


class ExperimentalDataCollector(DataCollector):
    """
    Users should implement these methods to collect data from their system directly.
    """

    def __init__(
        self,
        control_input_configuration: dict,
        treatment_input_configuration: dict,
        n_repeats: int = 1,
    ):
        super().__init__()
        self.control_input_configuration = control_input_configuration
        self.treatment_input_configuration = treatment_input_configuration
        self.n_repeats = n_repeats

    @abstractmethod
    def collect_data(self, **kwargs) -> pd.DataFrame:
        """
        Populate the dataframe with execution data.
        :return df: A pandas dataframe containing execution data for the system-under-test in both control and treatment
        executions.
        """
        control_df = self.run_system_with_input_configuration(
            self.control_input_configuration
        )
        treatment_df = self.run_system_with_input_configuration(
            self.treatment_input_configuration
        )
        return pd.concat([control_df, treatment_df], keys=["control", "treatment"])

    @abstractmethod
    def run_system_with_input_configuration(
        self, input_configuration: dict
    ) -> pd.DataFrame:
        """
        Run the system with a given input configuration and return the resulting execution data.
        :param input_configuration: A dictionary which maps a subset of inputs to values.
        :return df: A pandas dataframe containing execution data obtained by executing the system-under-test with the
        specified input configuration.
        """
        pass


class ObservationalDataCollector(DataCollector):
    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario

    def collect_data(self, csv_path: str) -> pd.DataFrame:
        """
        Read a csv containing execution data for the system-under-test into a pandas dataframe and filter to remove any
        data which does is invalid for the scenario-under-test. Data is invalid if it does not meet the constraints
        outlined in the scenario-under-test (Scenario).
        :param csv_path: Path to the csv containing execution data.
        :return scenario_execution_data_df: A pandas dataframe containing execution data that is valid for the
        scenario-under-test.
        """
        execution_data_df = pd.read_csv(csv_path)
        scenario_execution_data_df = self.filter_valid_data(execution_data_df)
        return scenario_execution_data_df

    def filter_valid_data(self, execution_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check is execution data is valid for the scenario-under-test. Data is invalid if it does not meet the
        constraints imposed in the scenario-under-test (Scenario).
        :param execution_data_df: A pandas dataframe containing execution data from the system-under-test.
        :return:
        """

        # Check positivity
        if not self.scenario.variables.issubset(execution_data_df.columns):
            missing_variables = self.scenario.variables - set(execution_data_df.columns)
            raise IndexError(
                f"Positivity violation: missing data for variables {missing_variables}."
            )

        # For each variable in the scenario, remove rows where a constraint is violated
        scenario_execution_data_df = execution_data_df.copy()
        for constraint in self.scenario.constraints:
            for variable in self.scenario.variables:
                scenario_execution_data_df = execution_data_df[
                    execution_data_df[variable].map(
                        lambda x: value_meets_constraint(x, constraint)
                    )
                ]

        return scenario_execution_data_df


# TODO Implement this
def value_meets_constraint(value: Union[int, float], constraint: ExprRef) -> bool:
    """
    Check that numerical value-specific constraints are met.
    :param value: The numerical value to check.
    :param constraint: The constraint to check the value against.
    :return: True or False depending on whether the value satisfies the constraint.
    """
    # if isinstance(constraint, AbsoluteValue):
    #     return value == constraint.value
    # elif isinstance(constraint, UniformDistribution):
    #     return constraint.min <= value <= constraint.max
    pass
