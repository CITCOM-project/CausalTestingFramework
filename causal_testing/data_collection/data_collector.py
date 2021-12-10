from abc import ABC, abstractmethod
from causal_testing.specification.causal_specification import Scenario

from typing import Union
import pandas as pd
import z3

import logging
logger = logging.getLogger(__name__)


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

    def collect_data(self, csv_path: str, **kwargs) -> pd.DataFrame:
        """
        Read a csv containing execution data for the system-under-test into a pandas dataframe and filter to remove any
        data which does is invalid for the scenario-under-test. Data is invalid if it does not meet the constraints
        outlined in the scenario-under-test (Scenario).
        :param csv_path: Path to the csv containing execution data.
        :return scenario_execution_data_df: A pandas dataframe containing execution data that is valid for the
        scenario-under-test.
        """
        execution_data_df = pd.read_csv(csv_path, **kwargs)
        scenario_execution_data_df = self.filter_valid_data(execution_data_df)
        return scenario_execution_data_df

    def filter_valid_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check is execution data is valid for the scenario-under-test. Data is invalid if it does not meet the
        constraints imposed in the scenario-under-test (Scenario).
        :param execution_data_df: A pandas dataframe containing execution data from the system-under-test.
        :return:
        """

        # Check positivity
        scenario_variables = set(self.scenario.variables)
        if not scenario_variables.issubset(data.columns):
            missing_variables = scenario_variables - set(data.columns)
            raise IndexError(
                f"Positivity violation: missing data for variables {missing_variables}."
            )

        # Check all variables declared in the modelling scenario
        # TODO: @andrewc19, does this have a name?
        if not set(data.columns).issubset(scenario_variables):
            missing_variables = set(data.columns) - set(variables)
            raise IndexError(
                f"Variables {missing_variables} not declared in the modelling scenario."
            )

        # For each row, does it satisfy the constraints?
        solver = z3.Solver()
        solver.add(self.scenario.constraints)
        sat = []
        for _, row in data.iterrows():
            solver.push()
            # Need to explicitly cast variables to their specified type. Z3 will not take e.g. np.int64 to be an int.
            model = [self.scenario.variables[var].z3 == self.scenario.variables[var].cast(row[var]) for var in data.columns]
            solver.add(model)
            sat.append(solver.check() == z3.sat)
            solver.pop()

        # Strip out rows which violate the constraints
        satisfying_data = data.copy()
        satisfying_data["sat"] = sat
        satisfying_data = satisfying_data.loc[satisfying_data["sat"]]
        satisfying_data = satisfying_data.drop("sat", axis=1)

        # How many rows did we drop?
        size_diff = len(data) - len(satisfying_data)
        if size_diff > 0:
            # TODO: Why does this print out many many times?
            logger.warn(f"Discarded {size_diff} values due to constraint violations.")
        return satisfying_data
