"""
This module implements the CausalTestingFramework class, which is the main interaction point for causal testing.
"""

import json
import logging
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_result import CausalTestResult, TestOutcome
from causal_testing.testing.data_adequacy import DataAdequacy

logger = logging.getLogger(__name__)

data_readers = {
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


def read_dataframe(file_path: str, content: bytes = None, **kwargs: dict) -> pd.DataFrame:
    """
    Read data into a dataframe.

    :param file_path: The path to the data.
    :param content: The bytes content of the file.
    :param kwargs: Keyword arguments to be passed to the `read_` function.

    :returns: The read-in DataFrame.
    """

    suffix = Path(file_path).suffix.lower()

    if suffix in data_readers:
        return data_readers[suffix](content if content is not None else file_path, **kwargs)
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

    def setup(
        self,
        dag_path: str = None,
        data_paths: list[str] = None,
        test_cases_path: str = None,
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
        if dag_path is not None:
            self.load_dag(dag_path, ignore_cycles)
        if data_paths is not None:
            self.load_data(data_paths, query, **kwargs)
        if test_cases_path is not None:
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

        if self.dag is None:
            raise ValueError("Please load DAG before attempting to load tests.")

        with open(test_cases_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        self.test_cases = [self.create_causal_test(test) for test in test_configs]

    def create_causal_test(self, test: dict) -> CausalTestCase:
        """
        Create causal test case from test configuration and base test.

        :param test: Dictionary containing test configuration parameters

        :return: CausalTestCase object
        :raises: ValueError if invalid estimator or configuration is provided
        """
        # Create the estimator with correct parameters
        estimator_map = {ff.name: ff for ff in entry_points(group="estimators")}

        if "estimator" not in test:
            raise ValueError("Test configuration must specify an estimator")

        estimator_class = test["estimator"].pop("name")
        if estimator_class not in estimator_map:
            raise ValueError(
                f"Unsupported estimator {estimator_class}. Supported: {sorted(estimator_map)}. "
                "If you have implemented a custom estimator, you will need to add this to your entrypoints via your "
                "pyproject.toml file."
            )

        estimator_class = estimator_map.get(estimator_class).load()
        test["estimator"] = estimator_class(**test["estimator"])

        # Create the expected effect with correct parameters
        effect_map = {ff.name: ff for ff in entry_points(group="causal_effects")}

        if "expected_causal_effect" not in test:
            raise ValueError("Test configuration must specify an expected causal effect.")

        effect_class = test["expected_causal_effect"].pop("name")
        if effect_class not in effect_map:
            raise ValueError(
                f"Unsupported causal effect {effect_class}. Supported: {sorted(effect_map)}. "
                "If you have implemented a custom causal effect, you will need to add this to your entrypoints via "
                "your pyproject.toml file."
            )
        effect_class = effect_map.get(effect_class).load()
        test["expected_causal_effect"] = effect_class(**test["expected_causal_effect"])

        if "result" in test:
            outcome = getattr(TestOutcome, test["result"]["outcome"]) if "outcome" in test["result"] else None
            effect_estimate = (
                EffectEstimate(**test["result"]["effect_estimate"]) if "effect_estimate" in test["result"] else None
            )
            adequacy = DataAdequacy(**test["result"]["adequacy"]) if "adequacy" in test["result"] else None

            test["result"] = CausalTestResult(outcome=outcome, effect_estimate=effect_estimate, adequacy=adequacy)

        return CausalTestCase(**test)

    def ready_to_run(self) -> bool:
        """
        Test whether framework is ready to run test cases.
        :returns: True if the DAG, data, and test cases are defined.
        """
        return all(x is not None for x in (self.test_cases, self.dag, self.df)) and bool(self.test_cases)

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

    def evaluate_dag(self, bootstrap_size: bool = 100, alpha: float = 0.05) -> pd.Series:
        """
        Calculate confidence intervals for how well a causal DAG fits a dataset by repeatedly resampling the dataset
        and executing the causal tests.
        Confidence intervals are then calcualted for how many tests pass, fail, or are inestimable.

        :param bootstrap_size: The number of bootstrap samples to use when calculating causal test adequacy
                               (defaults to 100)
        :param alpha: The significance level to use when calculating confidence intervals.
                      (defaults to 0.05).
        """
        self.run_tests(silent=True, adequacy=False)
        results = {
            test_outcome.name: len(
                [test for test in self.test_cases if test.result and test.result.outcome == test_outcome]
            )
            for test_outcome in TestOutcome
        }

        sample_results = []
        for sample_index in range(bootstrap_size):
            test_outcomes = {test_outcome: 0 for test_outcome in TestOutcome}
            for test_case in self.test_cases:
                if test_case.skip:
                    continue
                try:
                    effect_estimate = test_case.estimate_effect(
                        df=self.df.sample(len(self.df), replace=True, random_state=sample_index)
                    )
                except (np.linalg.LinAlgError, ValueError):
                    test_outcomes[TestOutcome.INESTIMABLE] += 1

                if effect_estimate:
                    if test_case.expected_causal_effect.apply(effect_estimate):
                        test_outcomes[TestOutcome.PASS] += 1
                    else:
                        test_outcomes[TestOutcome.FAIL] += 1
            sample_results.append(test_outcomes)

        sample_results = pd.DataFrame(sample_results)

        # Calculate the confidence interval of each column
        ci_low_inx = round((alpha / 2) * bootstrap_size)
        ci_high_inx = round(((1 - alpha) / 2) * bootstrap_size)
        for outcome in TestOutcome:
            data = sorted(sample_results[outcome])
            results[f"{outcome.name}_ci_low"] = data[ci_low_inx]
            results[f"{outcome.name}_ci_high"] = data[ci_high_inx]

        return pd.Series(results).sort_index()

    def save_results(self, output_path) -> list:
        """Save test results to JSON file in the expected format."""
        logger.info(f"Saving results to {output_path}")

        # Create parent directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([test.to_dict() for test in self.test_cases], f, indent=2)

        logger.info("Results saved successfully")

    def test_dataframe(self) -> pd.DataFrame:
        """
        :returns: The causal test cases as a dataframe. Nested objects such as results are indexed as, e.g.
        `result.outcome`.
        """
        return pd.json_normalize(map(lambda t: t.to_dict(), self.test_cases))
