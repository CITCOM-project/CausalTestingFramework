import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from causal_testing.__main__ import main

DAG_PATH = "tests/resources/data/dag.dot"
DATA_PATHS = ["tests/resources/data/data.csv"]
TEST_CASES_PATH = "tests/resources/data/tests.json"
OUTPUT_PATH = Path("results/results.json")
INCLUDE_EDGES_PATH = "tests/resources/data/include_edges.dot"
EXCLUDE_EDGES_PATH = "tests/resources/data/exclude_edges.dot"


@pytest.fixture(autouse=True)
def _remove_output_dir():
    yield
    if OUTPUT_PATH.parent.exists():
        shutil.rmtree(OUTPUT_PATH.parent)


def test_parse_args_test():
    with patch(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            str(DAG_PATH),
            "--data-paths",
            str(DATA_PATHS[0]),
            "--test-config",
            str(TEST_CASES_PATH),
            "--output",
            str(OUTPUT_PATH.parent / "main.json"),
        ],
    ):
        main()
        assert (OUTPUT_PATH.parent / "main.json").exists()


def test_parse_args_generate_and_test():
    with patch(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            str(DAG_PATH),
            "--data-paths",
            str(DATA_PATHS[0]),
            "--generate",
            "--output",
            str(OUTPUT_PATH.parent / "main.json"),
        ],
    ):
        main()
        assert (OUTPUT_PATH.parent / "main.json").exists()


def test_parse_args_adequacy():
    with patch(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            str(DAG_PATH),
            "--data-paths",
            str(DATA_PATHS[0]),
            "--test-config",
            str(TEST_CASES_PATH),
            "--output",
            str(OUTPUT_PATH.parent / "main.json"),
            "-A",
        ],
    ):
        main()
        with open(OUTPUT_PATH.parent / "main.json", encoding="utf-8") as f:
            log = json.load(f)
        executed_tests = [test for test in log if not test.get("skip", False)]
        assert all(test["result"].get("bootstrap_size", 100) == 100 for test in executed_tests)


def test_parse_args_bootstrap_size():
    with patch(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            str(DAG_PATH),
            "--data-paths",
            str(DATA_PATHS[0]),
            "--test-config",
            str(TEST_CASES_PATH),
            "--output",
            str(OUTPUT_PATH.parent / "main.json"),
            "-b",
            "50",
        ],
    ):
        main()
        with open(OUTPUT_PATH.parent / "main.json", encoding="utf-8") as f:
            log = json.load(f)
        executed_tests = [test for test in log if not test.get("skip", False)]
        assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)


def test_parse_args_bootstrap_size_explicit_adequacy():
    with patch(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            str(DAG_PATH),
            "--data-paths",
            str(DATA_PATHS[0]),
            "--test-config",
            str(TEST_CASES_PATH),
            "--output",
            str(OUTPUT_PATH.parent / "main.json"),
            "-A",
            "-b",
            "50",
        ],
    ):
        main()
        with open(OUTPUT_PATH.parent / "main.json", encoding="utf-8") as f:
            log = json.load(f)
        executed_tests = [test for test in log if not test.get("skip", False)]
        assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)


def test_parse_args_generate_then_test():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "generate",
                "--dag-path",
                str(DAG_PATH),
                "--data-paths",
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "tests.json"),
            ],
        ):
            main()
            assert os.path.exists(os.path.join(tmp, "tests.json"))
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "test",
                "--dag-path",
                str(DAG_PATH),
                "--data-paths",
                str(DATA_PATHS[0]),
                "--test-config",
                os.path.join(tmp, "tests.json"),
                "--output",
                str(OUTPUT_PATH.parent / "main.json"),
                "-A",
                "-b",
                "50",
            ],
        ):
            main()
            with open(OUTPUT_PATH.parent / "main.json", encoding="utf-8") as f:
                log = json.load(f)
            executed_tests = [test for test in log if not test.get("skip", False)]
            assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)


def test_parse_args_generation():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "generate",
                "--dag-path",
                str(DAG_PATH),
                "--data-paths",
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "tests.json"),
            ],
        ):
            main()
            assert os.path.exists(os.path.join(tmp, "tests.json"))


def test_parse_args_discover_malformed_argument():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "discover",
                "--technique",
                "HillClimberDiscovery",
                "--data-paths",
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "discovered_dag.dot"),
                "--include-edges",
                INCLUDE_EDGES_PATH,
                "--exclude-edges",
                EXCLUDE_EDGES_PATH,
                "--technique-kwargs",
                "max_iterations 10",
            ],
        ):
            with pytest.raises(
                ValueError,
                match=re.escape("Malformed argument max_iterations 10. Should be specified as `arg_name=arg_value`"),
            ):
                main()


def test_parse_args_discover():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "discover",
                "--technique",
                "HillClimberDiscovery",
                "--data-paths",
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "discovered_dag.dot"),
                "--include-edges",
                INCLUDE_EDGES_PATH,
                "--exclude-edges",
                EXCLUDE_EDGES_PATH,
                "--technique-kwargs",
                "max_iterations=10",
                "--variables",
                "test_input",
                "test_output",
            ],
        ):
            main()
            assert os.path.exists(os.path.join(tmp, "discovered_dag.dot"))


def test_parse_args_discover_invalid_technique():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "discover",
                "--technique",
                "Invalid",
                "--data-paths",
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "discovered_dag.dot"),
                "--include-edges",
                INCLUDE_EDGES_PATH,
                "--exclude-edges",
                EXCLUDE_EDGES_PATH,
                "--technique-kwargs",
                "max_iterations=10",
                "--variables",
                "test_input",
                "test_output",
            ],
        ):
            with pytest.raises(ValueError, match="^" + re.escape("Unsupported technique Invalid.")):
                main()


def test_parse_args_discover_drop_unnamed():
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
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "discovered_dag.dot"),
                "--technique-kwargs",
                "max_iterations=10",
            ],
        ):
            with pytest.warns(UserWarning, match=r"Dropping unnamed columns: \['Unnamed: 0'\]"):
                main()


def test_parse_args_evaluation_create_tests():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "evaluate",
                "--dag-path",
                str(DAG_PATH),
                "--data-paths",
                str(DATA_PATHS[0]),
                "--output",
                os.path.join(tmp, "results.csv"),
            ],
        ):
            main()
            assert os.path.exists(os.path.join(tmp, "results.csv"))


def test_parse_args_evaluation_read_tests():
    with tempfile.TemporaryDirectory() as tmp:
        with patch(
            "sys.argv",
            [
                "causal_testing",
                "evaluate",
                "--dag-path",
                str(DAG_PATH),
                "--data-paths",
                str(DATA_PATHS[0]),
                "--test-config",
                str(TEST_CASES_PATH),
                "--output",
                os.path.join(tmp, "results.csv"),
            ],
        ):
            main()
            assert os.path.exists(os.path.join(tmp, "results.csv"))
