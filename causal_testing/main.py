"""This module contains the classes that executes the Causal Testing Framework."""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Sequence

import pandas as pd

from .specification.causal_dag import CausalDAG
from .specification.scenario import Scenario
from .specification.variable import Input, Output
from .specification.causal_specification import CausalSpecification
from .testing.causal_test_case import CausalTestCase
from .testing.base_test_case import BaseTestCase
from .testing.causal_test_outcome import NoEffect, SomeEffect, Positive, Negative
from .testing.causal_test_result import CausalTestResult
from .estimation.linear_regression_estimator import LinearRegressionEstimator
from .estimation.logistic_regression_estimator import LogisticRegressionEstimator

logger = logging.getLogger(__name__)


@dataclass
class CausalTestingPaths:
    """
    Class for managing paths for causal testing inputs and outputs.

    :param dag_path: Path to the DAG definition file
    :param data_paths: List of paths to input data files
    :param test_config_path: Path to the test configuration file
    :param output_path: Path where test results will be written
    """

    dag_path: Path
    data_paths: List[Path]
    test_config_path: Path
    output_path: Path

    def __init__(
        self,
        dag_path: Union[str, Path],
        data_paths: List[Union[str, Path]],
        test_config_path: Union[str, Path],
        output_path: Union[str, Path],
    ):
        self.dag_path = Path(dag_path)
        self.data_paths = [Path(p) for p in data_paths]
        self.test_config_path = Path(test_config_path)
        self.output_path = Path(output_path)

    def validate_paths(self) -> None:
        """
        Validate existence of all input paths and writability of output path.

        :raises: FileNotFoundError if any required input file is missing.
        """

        if not self.dag_path.exists():
            raise FileNotFoundError(f"DAG file not found: {self.dag_path}")

        for data_path in self.data_paths:
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

        if not self.test_config_path.exists():
            raise FileNotFoundError(f"Test configuration file not found: {self.test_config_path}")

        if not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True)


class CausalTestingFramework:
    # pylint: disable=too-many-instance-attributes
    """
    Main class for running causal tests.

    :param paths: CausalTestingPaths object containing required file paths
    :param ignore_cycles: Flag to ignore cycles in the DAG
    :param query: Optional query string to filter the input dataframe

    """

    def __init__(self, paths: CausalTestingPaths, ignore_cycles: bool = False, query: Optional[str] = None):
        self.paths = paths
        self.ignore_cycles = ignore_cycles
        self.query = query

        # These will be populated during setup
        self.dag: Optional[CausalDAG] = None
        self.data: Optional[pd.DataFrame] = None
        self.variables: Dict[str, Any] = {"inputs": {}, "outputs": {}, "metas": {}}
        self.scenario: Optional[Scenario] = None
        self.causal_specification: Optional[CausalSpecification] = None
        self.test_cases: Optional[List[CausalTestCase]] = None

    def setup(self) -> None:
        """
        Set up the framework by loading DAG, runtime csv data, creating the scenario and causal specification.

        :raises: FileNotFoundError if required files are missing
        :raises: Exception if setup process fails
        """

        logger.info("Setting up Causal Testing Framework...")

        # Load and validate all paths
        self.paths.validate_paths()

        # Load DAG
        self.dag = self.load_dag()

        # Load data
        self.data = self.load_data(self.query)

        # Create variables from DAG
        self.create_variables()

        # Create scenario and specification
        self.create_scenario_and_specification()

        logger.info("Setup completed successfully")

    def load_dag(self) -> CausalDAG:
        """
        Load the causal DAG from the specified file path.

         :raises: Exception if DAG loading fails
        """
        logger.info(f"Loading DAG from {self.paths.dag_path}")
        try:
            dag = CausalDAG(str(self.paths.dag_path), ignore_cycles=self.ignore_cycles)
            logger.info(f"DAG loaded with {len(dag.graph.nodes)} nodes and {len(dag.graph.edges)} edges")
            return dag
        except Exception as e:
            logger.error(f"Failed to load DAG: {str(e)}")
            raise

    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load and combine all data sources with optional filtering.

        :param query: Optional pandas query string to filter the loaded data
        :return: Combined pandas DataFrame containing all loaded and filtered data
        :raises: Exception if data loading or query application fails
        """
        logger.info(f"Loading data from {len(self.paths.data_paths)} source(s)")

        try:
            dfs = [pd.read_csv(data_path) for data_path in self.paths.data_paths]
            data = pd.concat(dfs, axis=0, ignore_index=True)
            logger.info(f"Initial data shape: {data.shape}")

            if query:
                try:
                    logger.info(f"Attempting to apply query: '{query}'")
                    data = data.query(query)
                except Exception as e:
                    logger.error(f"Failed to apply query '{query}': {str(e)}")
                    raise

            return data
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def create_variables(self) -> None:
        """
        Create variable objects from DAG nodes based on their connectivity.


        """
        for node in self.dag.graph.nodes():
            dtype = self.data[node].dtype.type if node in self.data.columns else str

            # If node has no incoming edges, it's an input
            if self.dag.graph.in_degree(node) == 0:
                self.variables["inputs"][node] = Input(name=node, datatype=dtype)

            # If node has outgoing edges, it can be an input
            if self.dag.graph.out_degree(node) > 0:
                self.variables["inputs"][node] = Input(name=node, datatype=dtype)

            # If node has incoming edges, it can be an output
            if self.dag.graph.in_degree(node) > 0:
                self.variables["outputs"][node] = Output(name=node, datatype=dtype)

    def create_scenario_and_specification(self) -> None:
        """Create scenario and causal specification objects from loaded data.


        :raises: ValueError if scenario constraints filter out all data points
        """
        # Create scenario
        all_variables = list(self.variables["inputs"].values()) + list(self.variables["outputs"].values())
        self.scenario = Scenario(variables=all_variables)

        # Set up treatment variables
        self.scenario.setup_treatment_variables()

        # Apply scenario constraints to data
        self.apply_scenario_constraints()

        # Create causal specification
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.dag)

    def apply_scenario_constraints(self) -> None:
        """
        Apply scenario constraints to the loaded data.

        :raises: ValueError if all data points are filtered out by constraints
        """
        if not self.scenario.constraints:
            logger.info("No scenario constraints to apply")
            return

        original_rows = len(self.data)

        # Apply each constraint directly as a query string
        for constraint in self.scenario.constraints:
            try:
                self.data = self.data.query(str(constraint))
                logger.debug(f"Applied constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Failed to apply constraint '{constraint}': {str(e)}")

        filtered_rows = len(self.data)
        if filtered_rows < original_rows:
            logger.info(f"Scenario constraints filtered data from {original_rows} to {filtered_rows} rows")

        if filtered_rows == 0:
            raise ValueError("Scenario constraints filtered out all data points. Check your constraints and data.")

    def load_tests(self) -> None:
        """
        Load and prepare test configurations from file.


        :raises: Exception if test configuration loading fails
        """
        logger.info(f"Loading test configurations from {self.paths.test_config_path}")

        try:
            with open(self.paths.test_config_path, "r", encoding="utf-8") as f:
                test_configs = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test configurations: {str(e)}")
            raise

        self.test_cases = self.create_test_cases(test_configs)

    def create_base_test(self, test: dict) -> BaseTestCase:
        """
        Create base test case from test configuration.

        :param test: Dictionary containing test configuration parameters

        :return: BaseTestCase object
        :raises: KeyError if required variables are not found in inputs or outputs
        """
        treatment_name = test["mutations"][0]
        outcome_name = next(iter(test["expected_effect"].keys()))

        # Look for treatment variable in both inputs and outputs
        treatment_var = self.variables["inputs"].get(treatment_name) or self.variables["outputs"].get(treatment_name)
        if not treatment_var:
            raise KeyError(f"Treatment variable '{treatment_name}' not found in inputs or outputs")

        # Look for outcome variable in both inputs and outputs
        outcome_var = self.variables["inputs"].get(outcome_name) or self.variables["outputs"].get(outcome_name)
        if not outcome_var:
            raise KeyError(f"Outcome variable '{outcome_name}' not found in inputs or outputs")

        return BaseTestCase(
            treatment_variable=treatment_var, outcome_variable=outcome_var, effect=test.get("effect", "total")
        )

    def create_test_cases(self, test_configs: dict) -> List[CausalTestCase]:
        """Create test case objects from configuration dictionary.

        :param test_configs: Dictionary containing test configurations

        :return: List of CausalTestCase objects containing the initialised test cases
        :raises: KeyError if required variables are not found
        :raises: ValueError if invalid test configuration is provided
        """
        test_cases = []

        for test in test_configs.get("tests", []):
            if test.get("skip", False):
                continue

            # Create base test case
            base_test = self.create_base_test(test)

            # Create causal test case
            causal_test = self.create_causal_test(test, base_test)
            test_cases.append(causal_test)

        return test_cases

    def create_causal_test(self, test: dict, base_test: BaseTestCase) -> CausalTestCase:
        """
        Create causal test case from test configuration and base test.

        :param test: Dictionary containing test configuration parameters
        :param base_test: BaseTestCase object

        :return: CausalTestCase object
        :raises: ValueError if invalid estimator or configuration is provided
        """
        # Map effect string to effect class
        effect_map = {
            "NoEffect": NoEffect(),
            "SomeEffect": SomeEffect(),
            "Positive": Positive(),
            "Negative": Negative(),
        }

        # Map estimator string to estimator class
        estimator_map = {
            "LinearRegressionEstimator": LinearRegressionEstimator,
            "LogisticRegressionEstimator": LogisticRegressionEstimator,
        }

        if "estimator" not in test:
            raise ValueError("Test configuration must specify an estimator")

        # Get the estimator class
        estimator_class = estimator_map.get(test["estimator"])
        if estimator_class is None:
            raise ValueError(f"Unknown estimator: {test['estimator']}")

        # Create the estimator with correct parameters
        adjustment_set = self.causal_specification.causal_dag.identification(base_test)
        estimator = estimator_class(
            base_test_case=base_test,
            treatment_value=1.0,  # hardcode these for now
            control_value=0.0,
            adjustment_set=adjustment_set,
            df=self.data,
            effect_modifiers=None,
            formula=test.get("formula"),
            alpha=test.get("alpha", 0.05),
            query="",
        )

        # Get effect type and create expected effect
        effect_type = test["expected_effect"][base_test.outcome_variable.name]
        expected_effect = effect_map[effect_type]

        return CausalTestCase(
            base_test_case=base_test,
            expected_causal_effect=expected_effect,
            estimate_type=test.get("estimate_type", "ate"),
            estimate_params=test.get("estimate_params"),
            effect_modifier_configuration=test.get("effect_modifier_configuration"),
            estimator=estimator,
        )

    def run_tests(self) -> List[CausalTestResult]:
        """
        Run all test cases and return their results.

        :return: List of CausalTestResult objects
        :raises: ValueError if no tests are loaded
        :raises: Exception if test execution fails
        """
        logger.info("Running causal tests...")

        if not self.test_cases:
            raise ValueError("No tests loaded. Call load_tests() first.")

        results = []
        for test_case in self.test_cases:
            try:
                result = test_case.execute_test()
                results.append(result)
                logger.info(f"Test completed: {test_case}")
            except Exception as e:
                logger.error(f"Error running test {test_case}: {str(e)}")
                raise

        return results

    def save_results(self, results: List[CausalTestResult]) -> None:
        """Save test results to JSON file in the expected format."""
        logger.info(f"Saving results to {self.paths.output_path}")

        try:
            # Load original test configs to preserve test metadata
            with open(self.paths.test_config_path, "r", encoding="utf-8") as f:
                test_configs = json.load(f)

            # Combine test configs with their results
            json_results = []
            for test_config, test_case, result in zip(test_configs["tests"], self.test_cases, results):
                # Handle effect estimate - could be a Series or other format
                effect_estimate = result.test_value.value
                if isinstance(effect_estimate, pd.Series):
                    effect_estimate = effect_estimate.to_dict()

                # Handle confidence intervals - convert to list if needed
                ci_low = result.ci_low()
                ci_high = result.ci_high()
                if isinstance(ci_low, pd.Series):
                    ci_low = ci_low.tolist()
                if isinstance(ci_high, pd.Series):
                    ci_high = ci_high.tolist()

                # Determine if test failed based on expected vs actual effect
                test_failed = not test_case.expected_causal_effect.apply(result)

                output = {
                    "name": test_config["name"],
                    "estimate_type": test_config["estimate_type"],
                    "effect": test_config.get("effect", "direct"),
                    "mutations": test_config["mutations"],
                    "expected_effect": test_config["expected_effect"],
                    "formula": test_config.get("formula"),
                    "alpha": test_config.get("alpha", 0.05),
                    "skip": test_config.get("skip", False),
                    "failed": test_failed,
                    "result": {
                        "treatment": result.estimator.base_test_case.treatment_variable.name,
                        "outcome": result.estimator.base_test_case.outcome_variable.name,
                        "adjustment_set": list(result.adjustment_set) if result.adjustment_set else [],
                        "effect_measure": result.test_value.type,
                        "effect_estimate": effect_estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    },
                }
                json_results.append(output)

            # Save to file
            with open(self.paths.output_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2)

            logger.info("Results saved successfully")

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Causal Testing Framework")
    parser.add_argument("--dag_path", help="Path to the DAG file (.dot)", required=True)
    parser.add_argument("--data_paths", help="Paths to data files (.csv)", nargs="+", required=True)
    parser.add_argument("--test_config", help="Path to test configuration file (.json)", required=True)
    parser.add_argument("--output", help="Path for output file (.json)", required=True)
    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")
    parser.add_argument("--ignore-cycles", help="Ignore cycles in DAG", action="store_true")
    parser.add_argument("--query", help="Query string to filter data (e.g. 'age > 18')", type=str)

    return parser.parse_args(args)
