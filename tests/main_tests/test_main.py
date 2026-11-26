"""This module tests the command line interface."""

import json
import re

import pytest

from causal_testing.__main__ import main


def test_parse_args_test(tmp_path, monkeypatch, dag_path, data_path, test_cases_path):
    """Test that the `test` command runs the given test cases and writes the results file."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--test-config",
            test_cases_path,
            "--output",
            str(tmp_path / "main.json"),
        ],
    )
    main()
    assert (tmp_path / "main.json").exists()


def test_parse_args_generate_and_test(tmp_path, monkeypatch, dag_path, data_path):
    """Test that `test --generate` generates test cases from the DAG, runs them, and writes the results file."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--generate",
            "--output",
            str(tmp_path / "main.json"),
        ],
    )
    main()
    assert (tmp_path / "main.json").exists()


def test_parse_args_adequacy(tmp_path, monkeypatch, dag_path, data_path, test_cases_path):
    """Test that `-A` calculates adequacy using the default bootstrap size of 100."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--test-config",
            test_cases_path,
            "--output",
            str(tmp_path / "main.json"),
            "-A",
        ],
    )
    main()
    with open(tmp_path / "main.json", encoding="utf-8") as f:
        log = json.load(f)
    [executed_test] = [test for test in log if not test.get("skip", False)]
    assert executed_test["result"]["adequacy"]["bootstrap_size"] == 100


def test_parse_args_bootstrap_size(tmp_path, monkeypatch, dag_path, data_path, test_cases_path):
    """Test that `-b` sets the bootstrap size used to calculate adequacy."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--test-config",
            test_cases_path,
            "--output",
            str(tmp_path / "main.json"),
            "-b",
            "50",
        ],
    )
    main()
    with open(tmp_path / "main.json", encoding="utf-8") as f:
        log = json.load(f)
    [executed_test] = [test for test in log if not test.get("skip", False)]
    assert executed_test["result"]["adequacy"]["bootstrap_size"] == 50


def test_parse_args_bootstrap_size_explicit_adequacy(tmp_path, monkeypatch, dag_path, data_path, test_cases_path):
    """Test that `-b` sets the bootstrap size when adequacy is also explicitly requested with `-A`."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--test-config",
            test_cases_path,
            "--output",
            str(tmp_path / "main.json"),
            "-A",
            "-b",
            "50",
        ],
    )
    main()
    with open(tmp_path / "main.json", encoding="utf-8") as f:
        log = json.load(f)
    executed_tests = [test for test in log if not test.get("skip", False)]
    assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)


def test_parse_args_generate_then_test(tmp_path, monkeypatch, dag_path, data_path):
    """Test that test cases written by the `generate` command can be read and run by the `test` command."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "generate",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "tests.json"),
        ],
    )
    main()
    assert (tmp_path / "tests.json").exists()
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "test",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--test-config",
            str(tmp_path / "tests.json"),
            "--output",
            str(tmp_path / "main.json"),
            "-A",
            "-b",
            "50",
        ],
    )
    main()
    with open(tmp_path / "main.json", encoding="utf-8") as f:
        log = json.load(f)
    executed_tests = [test for test in log if not test.get("skip", False)]
    assert all(test["result"].get("bootstrap_size", 50) == 50 for test in executed_tests)


def test_parse_args_generation(tmp_path, monkeypatch, dag_path, data_path):
    """Test that the `generate` command writes the generated test cases to file."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "generate",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "tests.json"),
        ],
    )
    main()
    assert (tmp_path / "tests.json").exists()


def test_parse_args_discover_malformed_argument(
    tmp_path, monkeypatch, data_path, include_edges_path, exclude_edges_path
):
    """
    Test that `discover` raises an error when a technique keyword argument is not of the form `arg_name=arg_value`.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "discover",
            "--technique",
            "HillClimberDiscovery",
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "discovered_dag.dot"),
            "--include-edges",
            include_edges_path,
            "--exclude-edges",
            exclude_edges_path,
            "--technique-kwargs",
            "max_iterations 10",
        ],
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Malformed argument max_iterations 10. Should be specified as `arg_name=arg_value`"),
    ):
        main()


def test_parse_args_discover(tmp_path, monkeypatch, data_path, include_edges_path, exclude_edges_path):
    """Test that the `discover` command writes the discovered DAG to file."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "discover",
            "--technique",
            "HillClimberDiscovery",
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "discovered_dag.dot"),
            "--include-edges",
            include_edges_path,
            "--exclude-edges",
            exclude_edges_path,
            "--technique-kwargs",
            "max_iterations=10",
            "--variables",
            "test_input",
            "test_output",
        ],
    )
    main()
    assert (tmp_path / "discovered_dag.dot").exists()


def test_parse_args_discover_invalid_technique(
    tmp_path, monkeypatch, data_path, include_edges_path, exclude_edges_path
):
    """Test that `discover` raises an error for an unsupported discovery technique."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "discover",
            "--technique",
            "Invalid",
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "discovered_dag.dot"),
            "--include-edges",
            include_edges_path,
            "--exclude-edges",
            exclude_edges_path,
            "--technique-kwargs",
            "max_iterations=10",
            "--variables",
            "test_input",
            "test_output",
        ],
    )
    with pytest.raises(ValueError, match="^" + re.escape("Unsupported technique Invalid.")):
        main()


def test_parse_args_discover_drop_unnamed(tmp_path, monkeypatch, data_path):
    """
    Test that the user is warned of the dropping of unnamed columns.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "discover",
            "--technique",
            "HillClimberDiscovery",
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "discovered_dag.dot"),
            "--technique-kwargs",
            "max_iterations=10",
        ],
    )
    with pytest.warns(UserWarning, match=r"Dropping unnamed columns: \['Unnamed: 0'\]"):
        main()


def test_parse_args_evaluation_create_tests(tmp_path, monkeypatch, dag_path, data_path):
    """
    Test that the `evaluate` command generates test cases from the DAG when none are given and writes the results file.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "evaluate",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--output",
            str(tmp_path / "results.csv"),
        ],
    )
    main()
    assert (tmp_path / "results.csv").exists()


def test_parse_args_evaluation_read_tests(tmp_path, monkeypatch, dag_path, data_path, test_cases_path):
    """Test that the `evaluate` command uses the given test cases and writes the results file."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "causal_testing",
            "evaluate",
            "--dag-path",
            dag_path,
            "--data-paths",
            data_path,
            "--test-config",
            test_cases_path,
            "--output",
            str(tmp_path / "results.csv"),
        ],
    )
    main()
    assert (tmp_path / "results.csv").exists()
