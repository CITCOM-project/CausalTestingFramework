import pandas as pd
import z3
import logging
from abc import ABC, abstractmethod
from causal_testing.specification.causal_specification import Scenario

logger = logging.getLogger(__name__)


class DataCollector(ABC):

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    @abstractmethod
    def collect_data(self, **kwargs) -> pd.DataFrame:
        """
        Populate the dataframe with execution data.
        :return df: A pandas dataframe containing execution data for the system-under-test.
        """
        pass

    def filter_valid_data(self, data: pd.DataFrame, check_pos: bool = True,
                          check_scenario: bool = True) -> pd.DataFrame:
        """
        Check is execution data is valid for the scenario-under-test. Data is invalid if it does not meet the
        constraints imposed in the scenario-under-test (Scenario).
        :param data: A pandas dataframe containing execution data from the system-under-test.
        :param bool check_pos: Check the data for positivity (defaults to true).
        :param bool check_scenario: Make sure all data variables are defined in the scenario (defaults to true).
        :return:
        """

        # Check positivity
        scenario_variables = set(self.scenario.variables)

        if check_pos and not scenario_variables.issubset(data.columns):
            missing_variables = scenario_variables - set(data.columns)
            raise IndexError(f"Positivity violation: missing data for variables {missing_variables}.")

        if check_scenario and not set(data.columns).issubset(scenario_variables):
            missing_variables = set(data.columns) - scenario_variables
            raise IndexError(f"Variables {missing_variables} not declared in the modelling scenario.")

        # For each row, does it satisfy the constraints?
        solver = z3.Solver()
        for c in self.scenario.constraints:
            solver.assert_and_track(c, f"background: {c}")
        sat = []
        unsat_core = None
        for _, row in data.iterrows():
            solver.push()
            # Need to explicitly cast variables to their specified type. Z3 will not take e.g. np.int64 to be an int.
            model = [self.scenario.variables[var].z3 == self.scenario.variables[var].cast(row[var]) for var in
                     data.columns]
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
            logger.warning(f"Discarded {size_diff}/{len(data)} values due to constraint violations.\n"+
            f"For example{unsat_core}")
        return satisfying_data


class ExperimentalDataCollector(DataCollector):
    """
    Users should implement these methods to collect data from their system directly.
    """

    def __init__(self, scenario: Scenario, control_input_configuration: dict, treatment_input_configuration: dict,
                 n_repeats: int = 1):
        super().__init__(scenario)
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

        # Check runtime configs to make sure they don't violate constraints
        control_df = self.run_system_with_input_configuration(
            self.filter_valid_data(self.control_input_configuration, check_pos=False, check_scenario=False)
        )
        treatment_df = self.run_system_with_input_configuration(
            self.filter_valid_data(self.treatment_input_configuration, check_pos=False, check_scenario=False)
        )

        # Need to check final output too just in case we have constraints on output variables
        return self.filter_valid_data(pd.concat([control_df, treatment_df], keys=["control", "treatment"]))

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
    def __init__(self, scenario: Scenario, csv_path: str):
        super().__init__(scenario)
        self.csv_path = csv_path

    def collect_data(self, **kwargs) -> pd.DataFrame:
        """
        Read a csv containing execution data for the system-under-test into a pandas dataframe and filter to remove any
        data which is invalid for the scenario-under-test. Data is invalid if it does not meet the constraints
        outlined in the scenario-under-test (Scenario).

        :param scenario: Scenario for which the observational data is collected.

        :return: A pandas dataframe containing execution data that is valid for the scenario-under-test.
        :rtype: pd.DataFrame

        """

        execution_data_df = pd.read_csv(self.csv_path, **kwargs)
        scenario_execution_data_df = self.filter_valid_data(execution_data_df)
        return scenario_execution_data_df
