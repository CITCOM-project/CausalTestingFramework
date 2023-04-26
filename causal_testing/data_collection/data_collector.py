"""This module contains the DataCollector abstract class, as well as its concrete extensions: ExperimentalDataCollector
and ObservationalDataCollector"""

import logging
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
import z3

from causal_testing.specification.causal_specification import Scenario

logger = logging.getLogger(__name__)


class DataCollector(ABC):
    """A data collector is a mechanism which generates or collects data from a system for a given scenario."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    @abstractmethod
    def collect_data(self, **kwargs) -> pd.DataFrame:
        """
        Populate the dataframe with execution data.
        :return df: A pandas dataframe containing execution data for the system-under-test.
        """

    def filter_valid_data(self, data: pd.DataFrame, check_pos: bool = True) -> pd.DataFrame:
        """Check is execution data is valid for the scenario-under-test.

        Data is invalid if it does not meet the constraints specified in the scenario-under-test.

        :param data: A pandas dataframe containing execution data from the system-under-test.
        :param check_pos: Whether to check the data for positivity violations (defaults to true).
        :return satisfying_data: A pandas dataframe containing execution data that satisfy the constraints specified
        in the scenario-under-test.
        """

        # Check positivity
        scenario_variables = set(self.scenario.variables) - {x.name for x in self.scenario.hidden_variables()}

        if check_pos and not (scenario_variables - {x.name for x in self.scenario.hidden_variables()}).issubset(
            set(data.columns)
        ):
            missing_variables = scenario_variables - set(data.columns)
            raise IndexError(
                f"Missing columns: missing data for variables {missing_variables}. Should they be marked as hidden?"
            )

        # For each row, does it satisfy the constraints?
        solver = z3.Solver()
        for c in self.scenario.constraints:
            solver.assert_and_track(c, f"background: {c}")
        sat = []
        unsat_core = None
        for _, row in data.iterrows():
            solver.push()
            # Need to explicitly cast variables to their specified type. Z3 will not take e.g. np.int64 to be an int.
            model = [
                self.scenario.variables[var].z3
                == self.scenario.variables[var].z3_val(self.scenario.variables[var].z3, row[var])
                for var in self.scenario.variables
                if var in row and not pd.isnull(row[var])
            ]
            for c in model:
                solver.assert_and_track(c, f"model: {c}")
            check = solver.check()
            if check == z3.unsat and unsat_core is None:
                unsat_core = solver.unsat_core()
            sat.append(check == z3.sat)
            solver.pop()

        # Strip out rows which violate the constraints
        satisfying_data = data.copy()
        satisfying_data["sat"] = sat
        satisfying_data = satisfying_data.loc[satisfying_data["sat"]]
        satisfying_data = satisfying_data.drop("sat", axis=1)

        # How many rows did we drop?
        size_diff = len(data) - len(satisfying_data)
        if size_diff > 0:
            logger.warning(
                f"Discarded {size_diff}/{len(data)} values due to constraint violations.\n For example {unsat_core}",
            )
        return satisfying_data


class ExperimentalDataCollector(DataCollector):
    """A data collector that generates data directly by running the system-under-test in the desired conditions.

    Users should implement these methods to collect data from their system.
    """

    def __init__(
        self,
        scenario: Scenario,
        control_input_configuration: dict,
        treatment_input_configuration: dict,
        n_repeats: int = 1,
    ):
        super().__init__(scenario)
        self.control_input_configuration = control_input_configuration
        self.treatment_input_configuration = treatment_input_configuration
        self.n_repeats = n_repeats

    def collect_data(self, **kwargs) -> pd.DataFrame:
        """Run the system-under-test with control and treatment input configurations to obtain experimental data in
        which the causal effect of interest is isolated by design.

        :return: A pandas dataframe containing execution data for the system-under-test in both control and treatment
        executions.
        """
        control_results_df = self.run_system_with_input_configuration(self.control_input_configuration)
        control_results_df.rename(lambda x: f"control_{x}", inplace=True)
        treatment_results_df = self.run_system_with_input_configuration(self.treatment_input_configuration)
        treatment_results_df.rename(lambda x: f"treatment_{x}", inplace=True)
        results_df = pd.concat([control_results_df, treatment_results_df], ignore_index=False)
        return results_df

    @abstractmethod
    def run_system_with_input_configuration(self, input_configuration: dict) -> pd.DataFrame:
        """Run the system with a given input configuration and return the resulting execution data.

        :param input_configuration: A dictionary which maps a subset of inputs to values.
        :return: A pandas dataframe containing execution data obtained by executing the system-under-test with the
        specified input configuration.
        """


class ObservationalDataCollector(DataCollector):
    """A data collector that extracts data that is relevant to the specified scenario from a dataframe of execution
    data."""

    def __init__(self, scenario: Scenario, data: pd.DataFrame):
        super().__init__(scenario)
        self.data = data

    def collect_data(self, **kwargs) -> pd.DataFrame:
        """Read a pandas dataframe and filter to remove
        any data which is invalid for the scenario-under-test.

        Data is invalid if it does not meet the constraints outlined in the scenario-under-test (Scenario).

        :return: A pandas dataframe containing execution data that is valid for the scenario-under-test.
        """

        execution_data_df = self.data
        for meta in self.scenario.metas():
            if meta.name not in self.data:
                meta.populate(execution_data_df)
        scenario_execution_data_df = self.filter_valid_data(execution_data_df)
        for var_name, var in self.scenario.variables.items():
            if issubclass(var.datatype, Enum):
                scenario_execution_data_df[var_name] = [var.datatype(x) for x in scenario_execution_data_df[var_name]]
        return scenario_execution_data_df
