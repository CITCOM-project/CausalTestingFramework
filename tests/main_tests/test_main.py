import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from causal_testing.__main__ import main


class TestMain(unittest.TestCase):

    def setUp(self):
        self.dag_path = "tests/resources/data/dag.dot"
        self.data_paths = ["tests/resources/data/data.csv"]
        self.test_cases_path = "tests/resources/data/tests.json"
        self.output_path = Path("results/results.json")
        self.include_edges_path = "tests/resources/data/include_edges.dot"
        self.exclude_edges_path = "tests/resources/data/exclude_edges.dot"

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
                "-A",
            ],
        ):
            main()
            with open(self.output_path.parent / "main.json", encoding="utf-8") as f:
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
            with open(self.output_path.parent / "main.json", encoding="utf-8") as f:
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
                "-A",
                "-b",
                "50",
            ],
        ):
            main()
            with open(self.output_path.parent / "main.json", encoding="utf-8") as f:
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
                    "--data-paths",
                    str(self.data_paths[0]),
                    "--output",
                    os.path.join(tmp, "tests.json"),
                ],
            ):
                main()
                self.assertTrue(os.path.exists(os.path.join(tmp, "tests.json")))

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
                    "--output",
                    os.path.join(tmp, "discovered_dag.dot"),
                    "--include-edges",
                    self.include_edges_path,
                    "--exclude-edges",
                    self.exclude_edges_path,
                    "--technique-kwargs",
                    "max_iterations 10",
                ],
            ):
                with self.assertRaises(ValueError) as e:
                    main()
                    self.assertEqual(
                        e.message, "Malformed argument max_iterations. Should be specified as `arg_name=arg_value`"
                    )

    def test_parse_args_discover_malformed_argument(self):
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

    def test_parse_args_discover_invalid_technique(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "sys.argv",
                [
                    "causal_testing",
                    "discover",
                    "--technique",
                    "Invalid",
                    "--data-paths",
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
                with self.assertRaises(ValueError) as e:
                    main()
                    self.assertTrue(e.message.startswith("Unsupported technique Invalid."))

    def test_parse_args_discover_drop_unnamed(self):
        """
        Test that the user is warned of the dropping of unnamed columns.
        """
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
                    "--output",
                    os.path.join(tmp, "discovered_dag.dot"),
                    "--technique-kwargs",
                    "max_iterations=10",
                ],
            ):
                with self.assertWarnsRegex(UserWarning, r"Dropping unnamed columns: \['Unnamed: 0'\]"):
                    main()

    def test_parse_args_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "sys.argv",
                [
                    "causal_testing",
                    "evaluate",
                    "--dag-path",
                    str(self.dag_path),
                    "--data-paths",
                    str(self.data_paths[0]),
                    "--output",
                    os.path.join(tmp, "results.csv"),
                ],
            ):
                main()
                self.assertTrue(os.path.exists(os.path.join(tmp, "results.csv")))

    def tearDown(self):
        if self.output_path.parent.exists():
            shutil.rmtree(self.output_path.parent)
