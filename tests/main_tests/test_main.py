import unittest
from pathlib import Path
import tempfile
import os
from unittest.mock import patch
import shutil
import json
import pandas as pd

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.__main__ import main


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

    def test_load_dag_missing_node(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        framework.dag.add_node("missing")
        with self.assertRaises(ValueError):
            framework.create_variables()

    def test_load_tests_before_dag(self):
        framework = CausalTestingFramework()
        with self.assertRaises(ValueError):
            framework.load_test_cases_from_json(self.test_cases_path)

    def test_create_base_test_case_missing_treatment(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        with self.assertRaises(KeyError) as e:
            framework.create_base_test(
                {"treatment_variable": "missing", "expected_effect": {"test_outcome": "NoEffect"}}
            )
        self.assertEqual("\"Treatment variable 'missing' not found in inputs or outputs\"", str(e.exception))

    def test_create_base_test_case_missing_estimator(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        with self.assertRaises(ValueError) as e:
            framework.create_causal_test(
                {"treatment_variable": "test_input", "expected_effect": {"test_output": "NoEffect"}}
            )
        self.assertEqual("Test configuration must specify an estimator", str(e.exception))

    def test_create_test_case_invalid_estimator(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        with self.assertRaises(ValueError) as e:
            framework.create_causal_test(
                {
                    "treatment_variable": "test_input",
                    "expected_effect": {"test_output": "NoEffect"},
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
            "estimate_type": "coefficient",
            "expected_effect": {"test_output": "InvalidEffect"},
        }
        base_test_case = framework.create_base_test(test)
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
            "estimate_type": "coefficient",
            "expected_effect": {"test_output": "ExactValue"},
            "effect_kwargs": {"value": 4},
        }
        base_test_case = framework.create_base_test(test)
        test_case = framework.create_causal_test(test)
        self.assertEqual(test_case.expected_causal_effect.value, 4)

    def test_create_test_case_estimator_kwargs(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        test = {
            "name": "test1",
            "treatment_variable": "test_input",
            "estimator": "InstrumentalVariableEstimator",
            "estimate_type": "coefficient",
            "expected_effect": {"test_output": "SomeEffect"},
            "estimator_kwargs": {"instrument": "instrumental_variable"},
        }
        base_test_case = framework.create_base_test(test)
        test_case = framework.create_causal_test(test)
        self.assertEqual(test_case.estimator.instrument, "instrumental_variable")

    def test_create_base_test_case_missing_outcome(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        with self.assertRaises(KeyError) as e:
            framework.create_base_test({"treatment_variable": "test_input", "expected_effect": {"missing": "NoEffect"}})
        self.assertEqual("\"Outcome variable 'missing' not found in inputs or outputs\"", str(e.exception))

    def test_unloaded_tests(self):
        framework = CausalTestingFramework()
        with self.assertRaises(ValueError) as e:
            framework.run_tests()
        self.assertEqual("No tests to run.", str(e.exception))

    def test_ctf(self):
        framework = CausalTestingFramework()
        framework.setup(**self.paths)
        framework.run_tests()
        json_results = framework.save_results(self.output_path)

        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        self.assertEqual(len(json_results), len(test_configs["tests"]))

        result_index = 0
        for i, test_config in enumerate(test_configs["tests"]):
            result = json_results[i]

            if test_config.get("skip", False):
                self.assertEqual(result["skip"], True)
                self.assertEqual(result["passed"], None)
                self.assertEqual(result["result"]["status"], "skipped")
            else:
                test_case = framework.test_cases[result_index]
                result_index += 1

                test_passed = (
                    test_case.expected_causal_effect.apply(test_case.result.effect_estimate)
                    if test_case.result.effect_estimate is not None
                    else False
                )
                self.assertEqual(result["passed"], test_passed)

    def test_ctf_exception(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup(**self.paths, query="test_input < 0")

        with self.assertRaises(ValueError):
            framework.run_tests()

    def test_ctf_exception_silent(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup(**self.paths, query="test_input < 0")

        framework.run_tests(silent=True)
        json_results = framework.save_results(self.output_path)

        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

            non_skipped_configs = [t for t in test_configs["tests"] if not t.get("skip", False)]
            non_skipped_results = [r for r in json_results if not r.get("skip", False)]

            self.assertEqual(len(non_skipped_results), len(non_skipped_configs))

            for result in non_skipped_results:
                self.assertEqual(result["passed"], False)

    def test_parse_args(self):
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "test",
                "--dag-path",
                str(self.dag_path),
                "--data-paths",
                str(self.data_paths[0]),
                "--test-config",
                str(self.test_cases_path),
                "--output",
                str(self.output_path.parent / "main.json"),
            ],
        ):
            main()
            self.assertTrue((self.output_path.parent / "main.json").exists())

    def test_parse_args_adequacy(self):
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "test",
                "--dag-path",
                str(self.dag_path),
                "--data-paths",
                str(self.data_paths[0]),
                "--test-config",
                str(self.test_cases_path),
                "--output",
                str(self.output_path.parent / "main.json"),
                "-a",
            ],
        ):
            main()
            with open(self.output_path.parent / "main.json") as f:
                log = json.load(f)
            executed_tests = [test for test in log if not test.get("skip", False)]
            assert all(test["result"].get("bootstrap_size", 100) == 100 for test in executed_tests)

    def test_parse_args_bootstrap_size(self):
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "test",
                "--dag-path",
                str(self.dag_path),
                "--data-paths",
                str(self.data_paths[0]),
                "--test-config",
                str(self.test_cases_path),
                "--output",
                str(self.output_path.parent / "main.json"),
                "-b",
                "50",
            ],
        ):
            main()
            with open(self.output_path.parent / "main.json") as f:
                log = json.load(f)
            executed_tests = [test for test in log if not test.get("skip", False)]
            assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)

    def test_parse_args_bootstrap_size_explicit_adequacy(self):
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "test",
                "--dag-path",
                str(self.dag_path),
                "--data-paths",
                str(self.data_paths[0]),
                "--test-config",
                str(self.test_cases_path),
                "--output",
                str(self.output_path.parent / "main.json"),
                "-a",
                "-b",
                "50",
            ],
        ):
            main()
            with open(self.output_path.parent / "main.json") as f:
                log = json.load(f)
            executed_tests = [test for test in log if not test.get("skip", False)]
            assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)

    def test_parse_args_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "sys.argv",
                [
                    "causal_testing",
                    "generate",
                    "--dag-path",
                    str(self.dag_path),
                    "--output",
                    os.path.join(tmp, "tests.json"),
                ],
            ):
                main()
                self.assertTrue(os.path.exists(os.path.join(tmp, "tests.json")))

    def test_parse_args_generation_non_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "sys.argv",
                [
                    "causal_testing",
                    "generate",
                    "--dag-path",
                    str(self.dag_path),
                    "--output",
                    os.path.join(tmp, "tests_non_default.json"),
                    "--estimator",
                    "LogisticRegressionEstimator",
                    "--estimate-type",
                    "unit_odds_ratio",
                    "--effect-type",
                    "total",
                ],
            ):
                main()
                self.assertTrue(os.path.exists(os.path.join(tmp, "tests_non_default.json")))

    def test_parse_args_discover(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "sys.argv",
                [
                    "causal_testing",
                    "discover",
                    "--technique",
                    "HillClimberDiscovery",
                    "--data-paths",
                    str(self.data_paths[0]),
                    str(self.data_paths[0]),
                    "--output",
                    os.path.join(tmp, "discovered_dag.dot"),
                    "--include-edges",
                    self.include_edges_path,
                    "--exclude-edges",
                    self.exclude_edges_path,
                    "--technique-kwargs",
                    "max_iterations=10",
                    "--variables",
                    "test_input",
                    "test_output",
                ],
            ):
                main()
                self.assertTrue(os.path.exists(os.path.join(tmp, "discovered_dag.dot")))

    def tearDown(self):
        if self.output_path.parent.exists():
            shutil.rmtree(self.output_path.parent)
