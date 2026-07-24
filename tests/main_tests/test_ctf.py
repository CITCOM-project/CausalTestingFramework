import json
import unittest
from pathlib import Path

import pandas as pd

from causal_testing.causal_testing_framework import CausalTestingFramework


class TestCausalTestingFramework(unittest.TestCase):
    def setUp(self):
        self.dag_path = "tests/resources/data/dag.dot"
        self.data_paths = ["tests/resources/data/data.csv"]
        self.test_cases_path = "tests/resources/data/tests.json"
        self.output_path = Path("results/results.json")
        self.include_edges_path = "tests/resources/data/include_edges.dot"
        self.exclude_edges_path = "tests/resources/data/exclude_edges.dot"
        self.paths = {
            "dag_path": self.dag_path,
            "data_paths": self.data_paths,
            "test_cases_path": self.test_cases_path,
        }

    def test_load_data(self):
        csv_framework = CausalTestingFramework()
        csv_framework.load_data(self.data_paths)

        pqt_framework = CausalTestingFramework()
        pqt_framework.load_data([path.replace(".csv", ".pqt") for path in self.data_paths])
        pd.testing.assert_frame_equal(csv_framework.df, pqt_framework.df)

    def test_load_data_query(self):
        framework = CausalTestingFramework()
        framework.load_data(data_paths=self.data_paths)
        self.assertFalse((framework.df["test_input"] > 4).all())

        framework.load_data(data_paths=self.data_paths, query="test_input > 4")
        self.assertTrue((framework.df["test_input"] > 4).all())

    def test_load_data_invalid_extension(self):
        framework = CausalTestingFramework()
        with self.assertRaises(ValueError):
            framework.load_data("data.invalid")

    def test_load_tests_before_dag(self):
        framework = CausalTestingFramework()
        with self.assertRaises(ValueError):
            framework.load_test_cases_from_json(self.test_cases_path)

    def test_create_test_case_invalid_estimator(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        with self.assertRaises(ValueError) as e:
            framework.create_causal_test(
                {
                    "treatment_variable": "test_input",
                    "outcome_variable": "test_output",
                    "expected_effect": {"name": "NoEffect"},
                    "estimator": "InvalidEstimator",
                }
            )
        self.assertEqual(
            f"Unsupported estimator InvalidEstimator. Supported: ['CubicSplineEstimator', 'IPCWEstimator', 'InstrumentalVariableEstimator', 'LinearRegressionEstimator', 'LogisticRegressionEstimator', 'MultinomialRegressionEstimator']. "
            "If you have implemented a custom estimator, you will need to add this to your entrypoints via your "
            "pyproject.toml file.",
            str(e.exception),
        )

    def test_create_test_case_invalid_effect(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        test = {
            "name": "test1",
            "treatment_variable": "test_input",
            "estimator": "LinearRegressionEstimator",
            "effect_measure": "coefficient",
            "outcome_variable": "test_output",
            "expected_effect": {"name": "InvalidEffect"},
        }
        with self.assertRaises(ValueError) as e:
            framework.create_causal_test(test)
        self.assertEqual(
            f"Unsupported causal effect InvalidEffect. Supported: ['ExactValue', 'Negative', 'NoEffect', 'Positive', 'SomeEffect']. "
            "If you have implemented a custom causal effect, you will need to add this to your entrypoints via your "
            "pyproject.toml file.",
            str(e.exception),
        )

    def test_create_test_case_effect_kwargs(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        test = {
            "name": "test1",
            "treatment_variable": "test_input",
            "estimator": "LinearRegressionEstimator",
            "effect_measure": "coefficient",
            "outcome_variable": "test_output",
            "expected_effect": {"name": "ExactValue", "value": 4},
        }
        test_case = framework.create_causal_test(test)
        self.assertEqual(test_case.expected_causal_effect.value, 4)

    def test_create_test_case_estimator_kwargs(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        test = {
            "name": "test1",
            "treatment_variable": "test_input",
            "estimator": "InstrumentalVariableEstimator",
            "effect_measure": "coefficient",
            "outcome_variable": "test_output",
            "expected_effect": {"name": "SomeEffect"},
            "estimator_kwargs": {"instrument": "instrumental_variable"},
        }
        test_case = framework.create_causal_test(test)
        self.assertEqual(test_case.estimator.instrument, "instrumental_variable")

    def test_unloaded_tests(self):
        framework = CausalTestingFramework()
        with self.assertRaises(ValueError) as e:
            framework.run_tests()
        self.assertEqual("No tests to run.", str(e.exception))

    def test_ctf_exception(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup(**self.paths, query="test_input < 0")

        with self.assertRaises(ValueError):
            framework.run_tests()

    def test_ctf_exception_silent(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup(**self.paths, query="test_input < 0")
        framework.run_tests(silent=True)
        framework.save_results(self.output_path)

        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

            non_skipped_configs = [t for t in test_configs["tests"] if not t.get("skip", False)]
            non_skipped_results = [test.result for test in framework.test_cases if not test.skip]

            self.assertEqual(len(non_skipped_results), len(non_skipped_configs))

            for result in non_skipped_results:
                self.assertEqual(result.passed, False)

    def test_ctf_evaluate_dag(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup(**self.paths)
        results = framework.evaluate_dag()
        expected = pd.Series(
            {
                "PASS": 1,
                "FAIL": 0,
                "INESTIMABLE": 0,
                "PASS_ci_low": 0,
                "PASS_ci_high": 1,
                "FAIL_ci_low": 0,
                "FAIL_ci_high": 0,
                "INESTIMABLE_ci_low": 0,
                "INESTIMABLE_ci_high": 0,
            }
        ).sort_index()
        pd.testing.assert_series_equal(results, expected)
