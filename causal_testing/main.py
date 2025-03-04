"""This module contains the classes that executes the Causal Testing Framework."""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Sequence
from tqdm import tqdm

import pandas as pd
import numpy as np

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_outcome import NoEffect, SomeEffect, Positive, Negative
from causal_testing.testing.causal_test_result import CausalTestResult, TestValue
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator

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
        """
        logger.info(f"Loading DAG from {self.paths.dag_path}")
        dag = CausalDAG(str(self.paths.dag_path), ignore_cycles=self.ignore_cycles)
        logger.info(f"DAG loaded with {len(dag.graph.nodes)} nodes and {len(dag.graph.edges)} edges")
        return dag

    def _read_dataframe(self, data_path):
        if str(data_path).endswith(".csv"):
            return pd.read_csv(data_path)
        if str(data_path).endswith(".pqt"):
            return pd.read_parquet(data_path)
        raise ValueError(f"Invalid file type {data_path}. Can only read CSV (.csv) or parquet (.pqt) files.")

    def load_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Load and combine all data sources with optional filtering.

        :param query: Optional pandas query string to filter the loaded data
        :return: Combined pandas DataFrame containing all loaded and filtered data
        """
        logger.info(f"Loading data from {len(self.paths.data_paths)} source(s)")

        dfs = [self._read_dataframe(data_path) for data_path in self.paths.data_paths]
        data = pd.concat(dfs, axis=0, ignore_index=True)
        logger.info(f"Initial data shape: {data.shape}")

        if query:
            logger.info(f"Attempting to apply query: '{query}'")
            data = data.query(query)

        return data

    def create_variables(self) -> None:
        """
        Create variable objects from DAG nodes based on their connectivity.
        """
        for node_name, node_data in self.dag.graph.nodes(data=True):
            if node_name not in self.data.columns and not node_data.get("hidden", False):
                raise ValueError(f"Node {node_name} missing from data. Should it be marked as hidden?")

            dtype = self.data.dtypes.get(node_name)

            # If node has no incoming edges, it's an input
            if self.dag.graph.in_degree(node_name) == 0:
                self.variables["inputs"][node_name] = Input(name=node_name, datatype=dtype)

            # Otherwise it's an output
            if self.dag.graph.in_degree(node_name) > 0:
                self.variables["outputs"][node_name] = Output(name=node_name, datatype=dtype)

    def create_scenario_and_specification(self) -> None:
        """Create scenario and causal specification objects from loaded data."""
        # Create scenario
        all_variables = list(self.variables["inputs"].values()) + list(self.variables["outputs"].values())
        self.scenario = Scenario(variables=all_variables)

        # Set up treatment variables
        self.scenario.setup_treatment_variables()

        # Create causal specification
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.dag)

    def load_tests(self) -> None:
        """
        Load and prepare test configurations from file.
        """
        logger.info(f"Loading test configurations from {self.paths.test_config_path}")

        with open(self.paths.test_config_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        self.test_cases = self.create_test_cases(test_configs)

    def create_base_test(self, test: dict) -> BaseTestCase:
        """
        Create base test case from test configuration.

        :param test: Dictionary containing test configuration parameters

        :return: BaseTestCase object
        :raises: KeyError if required variables are not found in inputs or outputs
        """
        treatment_name = test["treatment_variable"]
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
        estimator = estimator_class(
            base_test_case=base_test,
            treatment_value=test.get("treatment_value"),
            control_value=test.get("control_value"),
            adjustment_set=test.get("adjustment_set", self.causal_specification.causal_dag.identification(base_test)),
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

    def run_tests_in_batches(self, batch_size: int = 100, silent: bool = False) -> List[CausalTestResult]:
        """
        Run tests in batches to reduce memory usage.

        :param batch_size: Number of tests to run in each batch
        :param silent: Whether to suppress errors
        :return: List of all test results
        :raises: ValueError if no tests are loaded
        """
        logger.info("Running causal tests in batches...")

        if not self.test_cases:
            raise ValueError("No tests loaded. Call load_tests() first.")

        num_tests = len(self.test_cases)
        num_batches = int(np.ceil(num_tests / batch_size))

        logger.info(f"Processing {num_tests} tests in {num_batches} batches of up to {batch_size} tests each")
        all_results = []
        with tqdm(total=num_tests, desc="Overall progress", mininterval=0.1) as progress:
            # Process each batch
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_tests)

                logger.info(f"Processing batch {batch_idx + 1} of {num_batches} (tests {start_idx} to {end_idx - 1})")

                # Get current batch of tests
                current_batch = self.test_cases[start_idx:end_idx]

                # Process the current batch
                batch_results = []
                for test_case in current_batch:
                    try:
                        result = test_case.execute_test()
                        batch_results.append(result)
                    except (TypeError, AttributeError) as e:
                        if not silent:
                            logger.error(f"Type or attribute error in test: {str(e)}")
                            raise
                        result = CausalTestResult(
                            estimator=test_case.estimator,
                            test_value=TestValue("Error", str(e)),
                        )
                        batch_results.append(result)

                    progress.update(1)

                all_results.extend(batch_results)

                logger.info(f"Completed batch {batch_idx + 1} of {num_batches}")

        logger.info(f"Completed processing all {len(all_results)} tests in {num_batches} batches")
        return all_results

    def run_tests(self, silent=False) -> List[CausalTestResult]:
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
        for test_case in tqdm(self.test_cases):
            try:
                result = test_case.execute_test()
                results.append(result)
                logger.info(f"Test completed: {test_case}")
            # pylint: disable=broad-exception-caught
            except Exception as e:
                if not silent:
                    logger.error(f"Error running test {test_case}: {str(e)}")
                    raise
                result = CausalTestResult(
                    estimator=test_case.estimator,
                    test_value=TestValue("Error", str(e)),
                )
                results.append(result)
                logger.info(f"Test errored: {test_case}")

        return results

    def save_results(self, results: List[CausalTestResult]) -> None:
        """Save test results to JSON file in the expected format."""
        logger.info(f"Saving results to {self.paths.output_path}")

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

            # Determine if test failed based on expected vs actual effect
            test_passed = test_case.expected_causal_effect.apply(result) if result.test_value.type != "Error" else False

            output = {
                "name": test_config["name"],
                "estimate_type": test_config["estimate_type"],
                "effect": test_config.get("effect", "direct"),
                "treatment_variable": test_config["treatment_variable"],
                "expected_effect": test_config["expected_effect"],
                "formula": test_config.get("formula"),
                "alpha": test_config.get("alpha", 0.05),
                "skip": test_config.get("skip", False),
                "passed": test_passed,
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
        return json_results


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Causal Testing Framework")
    parser.add_argument("-D", "--dag_path", help="Path to the DAG file (.dot)", required=True)
    parser.add_argument("-d", "--data_paths", help="Paths to data files (.csv)", nargs="+", required=True)
    parser.add_argument("-t", "--test_config", help="Path to test configuration file (.json)", required=True)
    parser.add_argument("-o", "--output", help="Path for output file (.json)", required=True)
    parser.add_argument("-v", "--verbose", help="Enable verbose logging", action="store_true", default=False)
    parser.add_argument("-i", "--ignore-cycles", help="Ignore cycles in DAG", action="store_true", default=False)
    parser.add_argument("-q", "--query", help="Query string to filter data (e.g. 'age > 18')", type=str)
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Do not crash on error. If set to true, errors are recorded as test results.",
        default=False,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Run tests in batches of the specified size (default: 0, which means no batching)",
    )

    return parser.parse_args(args)
