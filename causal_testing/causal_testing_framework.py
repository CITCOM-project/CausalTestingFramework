import json
import logging
from importlib.metadata import entry_points
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase

logger = logging.getLogger(__name__)


def read_dataframe(file_path: str, **kwargs: dict) -> pd.DataFrame:
    """
    Read data into a dataframe.

    :param file_path: The path to the data.
    :param kwargs: Keyword arguments to be passed to the `read_` function.

    :returns: The read-in DataFrame.
    """
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".html": pd.read_html,
        ".xml": pd.read_xml,
        ".feather": pd.read_feather,
        ".parquet": pd.read_parquet,
        ".pq": pd.read_parquet,
        ".pqt": pd.read_parquet,
        ".json": pd.read_json,
        ".stata": pd.read_stata,
    }

    suffix = Path(file_path).suffix.lower()

    if suffix in readers:
        return readers[suffix](file_path, **kwargs)
    raise ValueError(f"Unsupported file extension: '{suffix}'")


class CausalTestingFramework:
    """
    Main class for running causal tests.
    """

    def __init__(self, dag: CausalDAG = None, test_cases: list[CausalTestCase] = None, df: pd.DataFrame = None):
        self.dag = dag
        self.test_cases = test_cases
        self.df = df
        self.variables = {"inputs": {}, "outputs": {}}
        if self.dag is not None and self.df is not None:
            self.create_variables()

    def create_variables(self) -> None:
        """
        Create variable objects from DAG nodes based on their connectivity.
        """
        for node_name, node_data in self.dag.nodes(data=True):
            if node_name not in self.df.columns and not node_data.get("hidden", False):
                raise ValueError(f"Node {node_name} missing from data. Should it be marked as hidden?")

            dtype = self.df.dtypes.get(node_name)

            # If node has no incoming edges, it's an input
            if self.dag.in_degree(node_name) == 0:
                self.variables["inputs"][node_name] = Input(name=node_name, datatype=dtype)

            # Otherwise it's an output
            if self.dag.in_degree(node_name) > 0:
                self.variables["outputs"][node_name] = Output(name=node_name, datatype=dtype)

    def setup(
        self,
        dag_path: str,
        data_paths: list[str],
        test_cases_path: str,
        ignore_cycles: bool = False,
        query: str = None,
        **kwargs: dict,
    ):
        """
        Shortcut for loading in the DAG, data, and test cases.
        :param dag_path: Path to the DAG definition file.
        :param data_paths: List of paths to input data files.
        :param test_cases_path: Path to the test configuration file
        :param ignore_cycles: Whether to ignore cycles in the causal graph.
        NOTE: Setting this to True severely limits the testing that can be performed.
        :param query: Optional pandas query string to filter the loaded data
        :param kwargs: Keyword arguments to be passed to the `read_` function.
        """
        self.load_dag(dag_path, ignore_cycles)
        self.load_data(data_paths, query, **kwargs)
        self.load_test_cases_from_json(test_cases_path)

    def load_dag(self, dag_path: str, ignore_cycles: bool = False):
        """
        Load the causal DAG from the specified file path.

        :param dag_path: Path to the DAG definition file.
        :param ignore_cycles: Whether to ignore cycles in the causal graph.
                              NOTE: Setting this to True severely limits the testing that can be performed.
        """
        logger.info(f"Loading DAG from {dag_path}")
        self.dag = CausalDAG(dag_path, ignore_cycles=ignore_cycles)
        logger.info(f"DAG loaded with {len(self.dag.nodes)} nodes and {len(self.dag.edges)} edges")

    def load_data(self, data_paths: list[str], query: str = None, **kwargs: dict):
        """Load and combine all data sources with optional filtering.

        :param data_paths: List of paths to input data files.
        :param query: Optional pandas query string to filter the loaded data
        :param kwargs: Keyword arguments to be passed to the `read_` function.
        """
        logger.info(f"Loading data from {len(data_paths)} source(s)")

        data = pd.concat([read_dataframe(data_path, **kwargs) for data_path in data_paths], axis=0, ignore_index=True)
        logger.info(f"Initial data shape: {data.shape}")

        if query:
            logger.info(f"Attempting to apply query: '{query}'")
            data = data.query(query)

        self.df = data

    def load_test_cases_from_json(self, test_cases_path: str):
        """
        Load and prepare test configurations from JSON file.

        :param test_cases_path: Path to the test configuration file
        """
        logger.info(f"Loading test configurations from {test_cases_path}")

        if self.dag is None or self.df is None:
            raise ValueError("Please load DAG and data before attempting to load tests.")

        self.create_variables()

        with open(test_cases_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        test_cases = []

        for test in test_configs.get("tests", []):

            # Create causal test case
            causal_test = self.create_causal_test(test)
            test_cases.append(causal_test)

        self.test_cases = test_cases

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

    def create_causal_test(self, test: dict) -> CausalTestCase:
        """
        Create causal test case from test configuration and base test.

        :param test: Dictionary containing test configuration parameters

        :return: CausalTestCase object
        :raises: ValueError if invalid estimator or configuration is provided
        """
        estimator_map = {ff.name: ff for ff in entry_points(group="estimators")}
        effect_map = {ff.name: ff for ff in entry_points(group="causal_effects")}

        base_test = self.create_base_test(test)

        if "estimator" not in test:
            raise ValueError("Test configuration must specify an estimator")

        if test["estimator"] not in estimator_map:
            raise ValueError(
                f"Unsupported estimator {test['estimator']}. Supported: {sorted(estimator_map)}. "
                "If you have implemented a custom estimator, you will need to add this to your entrypoints via your "
                "pyproject.toml file."
            )

        # Create the estimator with correct parameters
        estimator_class = estimator_map.get(test["estimator"]).load()
        estimator_kwargs = test.get("estimator_kwargs", {})
        estimator = estimator_class(
            base_test_case=base_test,
            treatment_value=test.get("treatment_value"),
            control_value=test.get("control_value"),
            adjustment_set=test.get(
                "adjustment_set",
                self.dag.identification(base_test),
            ),
            alpha=test.get("alpha", 0.05),
            **estimator_kwargs,
        )

        # Get effect type and create expected effect
        effect_type = test["expected_effect"][base_test.outcome_variable.name]
        if effect_type not in effect_map:
            raise ValueError(
                f"Unsupported causal effect {effect_type}. Supported: {sorted(effect_map)}. "
                "If you have implemented a custom causal effect, you will need to add this to your entrypoints via "
                "your pyproject.toml file."
            )
        expected_effect = effect_map[effect_type].load()(**test.get("effect_kwargs", {}))

        return CausalTestCase(
            name=test.get("name"),
            query=test.get("query"),
            base_test_case=base_test,
            expected_causal_effect=expected_effect,
            estimate_type=test.get("estimate_type", "ate"),
            estimator=estimator,
            skip=test.get("skip", False),
        )

    def run_tests(self, silent: bool = False, adequacy: bool = False, bootstrap_size: int = 100):
        """
        Run all test cases and return their results.

        :param silent: Whether to suppress errors
        :param adequacy: Whether to calculate causal test adequacy (defaults to False)
        :param bootstrap_size: The number of bootstrap samples to use when calculating causal test adequacy
                               (defaults to 100)

        :raises: ValueError if no tests are loaded
        :raises: Exception if test execution fails
        """
        logger.info("Running causal tests...")

        if not self.test_cases:
            raise ValueError("No tests to run.")

        for test_case in tqdm(self.test_cases):
            test_case.execute_test(
                self.df, suppress_estimation_errors=silent, adequacy=adequacy, bootstrap_size=bootstrap_size
            )

    def save_results(self, output_path) -> list:
        """Save test results to JSON file in the expected format."""
        logger.info(f"Saving results to {output_path}")

        # Create parent directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        json_results = []
        result_index = 0

        for test_case in self.test_cases:
            # Create a base output first of common entries
            base_output = {
                "name": test_case.name,
                "estimate_type": test_case.estimate_type,
                "effect": test_case.base_test_case.effect,
                "treatment_variable": test_case.base_test_case.treatment_variable.name,
                "expected_effect": test_case.expected_causal_effect.__class__.__name__,
                "alpha": test_case.estimator.alpha,
            }
            if test_case.skip:
                # Include those skipped test entry without execution results
                output = {
                    **base_output,
                    "formula": result.estimator.formula if hasattr(result.estimator, "formula") else None,
                    "skip": True,
                    "passed": None,
                    "result": {
                        "status": "skipped",
                        "reason": "Test marked as skip:true in the causal test config file.",
                    },
                }
            else:
                result = test_case.result
                result_index += 1

                test_passed = (
                    test_case.expected_causal_effect.apply(result) if result.effect_estimate is not None else False
                )

                output = {
                    **base_output,
                    "formula": result.estimator.formula if hasattr(result.estimator, "formula") else None,
                    "skip": False,
                    "passed": test_passed,
                    "result": (
                        {
                            "treatment": result.estimator.base_test_case.treatment_variable.name,
                            "outcome": result.estimator.base_test_case.outcome_variable.name,
                            "adjustment_set": list(result.adjustment_set) if result.adjustment_set else [],
                        }
                        | result.effect_estimate.to_dict()
                        | (result.adequacy.to_dict() if result.adequacy else {})
                        if result.effect_estimate
                        else {"status": "error", "reason": result.error_message}
                    ),
                }

            json_results.append(output)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)

        logger.info("Results saved successfully")
        return json_results
