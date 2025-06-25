import unittest
from pathlib import Path
import tempfile
import os

import shutil
import json
import pandas as pd
from unittest.mock import patch

from causal_testing.main import CausalTestingPaths, CausalTestingFramework
from causal_testing.__main__ import main


class TestCausalTestingPaths(unittest.TestCase):

    def setUp(self):
        self.dag_path = "tests/resources/data/dag.dot"
        self.data_paths = ["tests/resources/data/data.csv"]
        self.test_config_path = "tests/resources/data/tests.json"
        self.output_path = Path("results/results.json")

    def test_missing_dag(self):
        with self.assertRaises(FileNotFoundError) as e:
            CausalTestingPaths("missing.dot", self.data_paths, self.test_config_path, self.output_path).validate_paths()
        self.assertEqual("DAG file not found: missing.dot", str(e.exception))

    def test_missing_data(self):
        with self.assertRaises(FileNotFoundError) as e:
            CausalTestingPaths(self.dag_path, ["missing.csv"], self.test_config_path, self.output_path).validate_paths()
        self.assertEqual("Data file not found: missing.csv", str(e.exception))

    def test_missing_tests(self):
        with self.assertRaises(FileNotFoundError) as e:
            CausalTestingPaths(self.dag_path, self.data_paths, "missing.json", self.output_path).validate_paths()
        self.assertEqual("Test configuration file not found: missing.json", str(e.exception))

    def test_output_file_created(self):
        self.assertFalse(self.output_path.parent.exists())
        CausalTestingPaths(self.dag_path, self.data_paths, self.test_config_path, self.output_path).validate_paths()
        self.assertTrue(self.output_path.parent.exists())

    def tearDown(self):
        if self.output_path.parent.exists():
            shutil.rmtree(self.output_path.parent)


class TestCausalTestingFramework(unittest.TestCase):
    def setUp(self):
        self.dag_path = "tests/resources/data/dag.dot"
        self.data_paths = ["tests/resources/data/data.csv"]
        self.test_config_path = "tests/resources/data/tests.json"
        self.output_path = Path("results/results.json")
        self.paths = CausalTestingPaths(
            dag_path=self.dag_path,
            data_paths=self.data_paths,
            test_config_path=self.test_config_path,
            output_path=self.output_path,
        )

    def test_load_data(self):
        csv_framework = CausalTestingFramework(self.paths)
        csv_df = csv_framework.load_data()

        pqt_framework = CausalTestingFramework(
            CausalTestingPaths(
                dag_path=self.dag_path,
                data_paths=["tests/resources/data/data.pqt"],
                test_config_path=self.test_config_path,
                output_path=self.output_path,
            )
        )
        pqt_df = pqt_framework.load_data()
        pd.testing.assert_frame_equal(csv_df, pqt_df)

    def test_load_data_invalid(self):
        framework = CausalTestingFramework(
            CausalTestingPaths(
                dag_path=self.dag_path,
                data_paths=[self.dag_path],
                test_config_path=self.test_config_path,
                output_path=self.output_path,
            )
        )
        with self.assertRaises(ValueError):
            framework.load_data()

    def test_load_data_query(self):
        framework = CausalTestingFramework(self.paths)
        self.assertFalse((framework.load_data()["test_input"] > 4).all())
        self.assertTrue((framework.load_data("test_input > 4")["test_input"] > 4).all())

    def test_load_dag_missing_node(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()
        framework.dag.graph.add_node("missing")
        with self.assertRaises(ValueError):
            framework.create_variables()

    def test_create_base_test_case_missing_treatment(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()
        with self.assertRaises(KeyError) as e:
            framework.create_base_test(
                {"treatment_variable": "missing", "expected_effect": {"test_outcome": "NoEffect"}}
            )
        self.assertEqual("\"Treatment variable 'missing' not found in inputs or outputs\"", str(e.exception))

    def test_create_base_test_case_missing_estimator(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()
        with self.assertRaises(ValueError) as e:
            framework.create_causal_test({}, None)
        self.assertEqual("Test configuration must specify an estimator", str(e.exception))

    def test_create_base_test_case_invalid_estimator(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()
        with self.assertRaises(ValueError) as e:
            framework.create_causal_test({"estimator": "InvalidEstimator"}, None)
        self.assertEqual("Unknown estimator: InvalidEstimator", str(e.exception))

    def test_create_base_test_case_missing_outcome(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()
        with self.assertRaises(KeyError) as e:
            framework.create_base_test({"treatment_variable": "test_input", "expected_effect": {"missing": "NoEffect"}})
        self.assertEqual("\"Outcome variable 'missing' not found in inputs or outputs\"", str(e.exception))

    def test_ctf(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()

        # Load and run tests
        framework.load_tests()
        results = framework.run_tests()

        # Save results
        framework.save_results(results)

        with open(self.test_config_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        tests_passed = [
            test_case.expected_causal_effect.apply(result) if result.test_value.type != "Error" else False
            for test_config, test_case, result in zip(test_configs["tests"], framework.test_cases, results)
        ]

        self.assertEqual(tests_passed, [True])

    def test_ctf_batches(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()

        # Load and run tests
        framework.load_tests()

        output_files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, results in enumerate(framework.run_tests_in_batches()):
                temp_file_path = os.path.join(tmpdir, f"output_{i}.json")
                framework.save_results(results, temp_file_path)
                output_files.append(temp_file_path)
                del results

            # Now stitch the results together from the temporary files
            all_results = []
            for file_path in output_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))

        self.assertEqual([result["passed"] for result in all_results], [True])

    def test_ctf_batches_exception_silent(self):
        framework = CausalTestingFramework(self.paths, query="test_input < 0")
        framework.setup()

        # Load and run tests
        framework.load_tests()

        output_files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, results in enumerate(framework.run_tests_in_batches(silent=True)):
                temp_file_path = os.path.join(tmpdir, f"output_{i}.json")
                framework.save_results(results, temp_file_path)
                output_files.append(temp_file_path)
                del results

            # Now stitch the results together from the temporary files
            all_results = []
            for file_path in output_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))

        self.assertEqual([result["passed"] for result in all_results], [False])
        self.assertEqual([result["result"]["effect_measure"] for result in all_results], ["Error"])

    def test_ctf_batches_exception(self):
        framework = CausalTestingFramework(self.paths, query="test_input < 0")
        framework.setup()

        # Load and run tests
        framework.load_tests()
        with self.assertRaises(ValueError):
            list(framework.run_tests_in_batches())

    def test_ctf_batches_matches_run_tests(self):
        # Run the tests normally
        framework = CausalTestingFramework(self.paths)
        framework.setup()
        framework.load_tests()
        normale_results = framework.run_tests()

        # Run the tests in batches
        output_files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, results in enumerate(framework.run_tests_in_batches()):
                temp_file_path = os.path.join(tmpdir, f"output_{i}.json")
                framework.save_results(results, temp_file_path)
                output_files.append(temp_file_path)
                del results

            # Now stitch the results together from the temporary files
            all_results = []
            for file_path in output_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))

        with tempfile.TemporaryDirectory() as tmpdir:
            normal_output = os.path.join(tmpdir, f"normal.json")
            framework.save_results(normale_results, normal_output)
            with open(normal_output) as f:
                normal_results = json.load(f)

            batch_output = os.path.join(tmpdir, f"batch.json")
            with open(batch_output, "w") as f:
                json.dump(all_results, f)
            with open(batch_output) as f:
                batch_results = json.load(f)
            self.assertEqual(normal_results, batch_results)

    def test_global_query(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()

        query_framework = CausalTestingFramework(self.paths, query="test_input > 0")
        query_framework.setup()

        self.assertTrue(len(query_framework.data) > 0)
        self.assertTrue((query_framework.data["test_input"] > 0).all())

        with open(self.test_config_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        test_config = test_configs["tests"][0].copy()
        if "query" in test_config:
            del test_config["query"]

        base_test = query_framework.create_base_test(test_config)
        causal_test = query_framework.create_causal_test(test_config, base_test)

        self.assertTrue((causal_test.estimator.df["test_input"] > 0).all())

        query_framework.create_variables()
        query_framework.create_scenario_and_specification()

        self.assertIsNotNone(query_framework.scenario)
        self.assertIsNotNone(query_framework.causal_specification)

    def test_test_specific_query(self):
        framework = CausalTestingFramework(self.paths)
        framework.setup()

        with open(self.test_config_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        test_config = test_configs["tests"][0].copy()
        test_config["query"] = "test_input > 0"

        base_test = framework.create_base_test(test_config)
        causal_test = framework.create_causal_test(test_config, base_test)

        self.assertTrue(len(causal_test.estimator.df) > 0)
        self.assertTrue((causal_test.estimator.df["test_input"] > 0).all())

    def test_combined_queries(self):
        global_framework = CausalTestingFramework(self.paths, query="test_input > 0")
        global_framework.setup()

        with open(self.test_config_path, "r", encoding="utf-8") as f:
            test_configs = json.load(f)

        test_config = test_configs["tests"][0].copy()
        test_config["query"] = "test_output > 0"

        base_test = global_framework.create_base_test(test_config)
        causal_test = global_framework.create_causal_test(test_config, base_test)

        self.assertTrue(len(causal_test.estimator.df) > 0)
        self.assertTrue((causal_test.estimator.df["test_input"] > 0).all())
        self.assertTrue((causal_test.estimator.df["test_output"] > 0).all())

    def test_parse_args(self):
        with unittest.mock.patch(
            "sys.argv",
            [
                "causal_testing",
                "--dag_path",
                str(self.dag_path),
                "--data_paths",
                str(self.data_paths[0]),
                "--test_config",
                str(self.test_config_path),
                "--output",
                str(self.output_path.parent / "main.json"),
            ],
        ):
            main()
            self.assertTrue((self.output_path.parent / "main.json").exists())

    def test_parse_args_batches(self):
        with unittest.mock.patch(
            "sys.argv",
            [
                "causal_testing",
                "--dag_path",
                str(self.dag_path),
                "--data_paths",
                str(self.data_paths[0]),
                "--test_config",
                str(self.test_config_path),
                "--output",
                str(self.output_path.parent / "main_batch.json"),
                "--batch-size",
                "5",
            ],
        ):
            main()
            self.assertTrue((self.output_path.parent / "main_batch.json").exists())

    def test_parse_args_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch(
                "sys.argv",
                [
                    "causal_testing",
                    "--generate",
                    "--dag_path",
                    str(self.dag_path),
                    "--output",
                    os.path.join(tmp, "tests.json"),
                ],
            ):
                main()
                self.assertTrue(os.path.exists(os.path.join(tmp, "tests.json")))

    def tearDown(self):
        if self.output_path.parent.exists():
            shutil.rmtree(self.output_path.parent)
