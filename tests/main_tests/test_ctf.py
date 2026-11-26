"""This module tests the CausalTestingFramework class."""

import json
import re

import pandas as pd
import pytest

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.specification.causal_dag import CausalDAG


@pytest.fixture(name="data_paths")
def _data_paths(data_path):
    """
    The example data path, as a list.
    """
    return [data_path]


@pytest.fixture(name="paths")
def _paths(dag_path, data_paths, test_cases_path):
    """
    Keyword arguments for `CausalTestingFramework.setup`.
    """
    return {
        "dag_path": dag_path,
        "data_paths": data_paths,
        "test_cases_path": test_cases_path,
    }


def test_load_data(data_paths):
    """Test that CSV and parquet versions of the same data load to identical dataframes."""
    csv_framework = CausalTestingFramework()
    csv_framework.load_data(data_paths)

    pqt_framework = CausalTestingFramework()
    pqt_framework.load_data([path.replace(".csv", ".pqt") for path in data_paths])
    pd.testing.assert_frame_equal(csv_framework.df, pqt_framework.df)


def test_load_data_query(data_paths):
    """Test that a query passed to load_data filters the loaded data."""
    framework = CausalTestingFramework()
    framework.load_data(data_paths=data_paths)
    assert not (framework.df["test_input"] > 4).all()

    framework.load_data(data_paths=data_paths, query="test_input > 4")
    assert (framework.df["test_input"] > 4).all()


def test_load_data_invalid_extension():
    """Test that load_data raises an error for an unsupported file extension."""
    framework = CausalTestingFramework()
    with pytest.raises(ValueError, match=re.escape("Unsupported file extension: '.invalid'")):
        framework.load_data(["data.invalid"])


def test_load_tests_before_dag(test_cases_path):
    """Test that loading test cases before the DAG raises an error."""
    framework = CausalTestingFramework()
    with pytest.raises(ValueError, match=re.escape("Please load DAG before attempting to load tests.")):
        framework.load_test_cases_from_json(test_cases_path)


def test_create_test_case_invalid_estimator(paths):
    """Test that creating a test case with an unknown estimator raises an error listing the supported estimators."""
    framework = CausalTestingFramework()
    framework.setup(**paths)
    message = (
        "Unsupported estimator InvalidEstimator. Supported: ['CubicSplineEstimator', 'IPCWEstimator', "
        "'InstrumentalVariableEstimator', 'LinearRegressionEstimator', 'LogisticRegressionEstimator', "
        "'MultinomialRegressionEstimator']. "
        "If you have implemented a custom estimator, you will need to add this to your entrypoints via your "
        "pyproject.toml file."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        framework.create_causal_test(
            {
                "treatment_variable": "test_input",
                "outcome_variable": "test_output",
                "expected_causal_effect": {"name": "NoEffect"},
                "estimator": {"name": "InvalidEstimator"},
            }
        )


def test_create_test_case_no_estimator(paths):
    """Test that creating a test case without an estimator raises an error."""
    framework = CausalTestingFramework()
    framework.setup(**paths)
    with pytest.raises(ValueError, match=re.escape("Test configuration must specify an `estimator`.")):
        framework.create_causal_test(
            {
                "treatment_variable": "test_input",
                "outcome_variable": "test_output",
                "expected_causal_effect": {"name": "NoEffect"},
            }
        )


def test_create_test_case_no_expected_causal_effect(dag_path, data_paths):
    """Test that creating a test case without an expected causal effect raises an error."""
    framework = CausalTestingFramework()
    framework.load_dag(dag_path)
    framework.load_data(data_paths)
    test = {
        "name": "test1",
        "estimator": {
            "name": "LinearRegressionEstimator",
            "treatment_variable": "test_input",
            "outcome_variable": "test_output",
            "adjustment_set": [],
        },
        "effect_measure": "coefficient",
    }
    with pytest.raises(ValueError, match=re.escape("Test configuration must specify an `expected_causal_effect`.")):
        framework.create_causal_test(test)


def test_create_test_case_invalid_effect(dag_path, data_paths):
    """Test that creating a test case with an unknown causal effect raises an error listing the supported effects."""
    framework = CausalTestingFramework()
    framework.load_dag(dag_path)
    framework.load_data(data_paths)
    test = {
        "name": "test1",
        "estimator": {
            "name": "LinearRegressionEstimator",
            "treatment_variable": "test_input",
            "outcome_variable": "test_output",
            "adjustment_set": [],
        },
        "effect_measure": "coefficient",
        "expected_causal_effect": {"name": "InvalidEffect"},
    }
    message = (
        "Unsupported causal effect InvalidEffect. "
        "Supported: ['ExactValue', 'Negative', 'NoEffect', 'Positive', 'SomeEffect']. "
        "If you have implemented a custom causal effect, you will need to add this to your entrypoints via your "
        "pyproject.toml file."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        framework.create_causal_test(test)


def test_create_test_case_effect_kwargs(dag_path, data_paths):
    """Test that extra keys in the expected causal effect configuration are passed to its constructor."""
    framework = CausalTestingFramework()
    framework.load_dag(dag_path)
    framework.load_data(data_paths)
    test = {
        "name": "test1",
        "estimator": {
            "name": "LinearRegressionEstimator",
            "treatment_variable": "test_input",
            "outcome_variable": "test_output",
            "adjustment_set": [],
        },
        "effect_measure": "coefficient",
        "expected_causal_effect": {"name": "ExactValue", "value": 4},
    }
    test_case = framework.create_causal_test(test)
    assert test_case.expected_causal_effect.value == 4


def test_create_test_case_estimator_kwargs(dag_path, data_paths):
    """Test that extra keys in the estimator configuration are passed to its constructor."""
    framework = CausalTestingFramework()
    framework.load_dag(dag_path)
    framework.load_data(data_paths)
    test = {
        "name": "test1",
        "estimator": {
            "name": "InstrumentalVariableEstimator",
            "treatment_variable": "test_input",
            "outcome_variable": "test_output",
            "instrument": "instrumental_variable",
        },
        "effect_measure": "coefficient",
        "expected_causal_effect": {"name": "SomeEffect"},
    }
    test_case = framework.create_causal_test(test)
    assert test_case.estimator.instrument == "instrumental_variable"


def test_unloaded_tests():
    """Test that running tests before any are loaded raises an error."""
    framework = CausalTestingFramework()
    with pytest.raises(ValueError, match=re.escape("No tests to run.")):
        framework.run_tests()


def test_ctf_exception(paths):
    """Test that an error during test execution is raised when not running silently."""
    framework = CausalTestingFramework(paths)
    framework.setup(**paths, query="test_input < 0")

    with pytest.raises(
        ValueError, match=re.escape("zero-size array to reduction operation maximum which has no identity")
    ):
        framework.run_tests()


def test_ctf_exception_silent(tmp_path, test_cases_path, paths):
    """
    Test that, when running silently, errors during test execution are caught and every non-skipped test is recorded as
    failing.
    """
    framework = CausalTestingFramework(paths)
    framework.setup(**paths, query="test_input < 0")
    framework.run_tests(silent=True)
    framework.save_results(tmp_path / "results.json")

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_configs = json.load(f)

        non_skipped_configs = [t for t in test_configs if not t.get("skip", False)]
        non_skipped_results = [test.result for test in framework.test_cases if not test.skip]

        assert len(non_skipped_results) == len(non_skipped_configs)

        for result in non_skipped_results:
            assert not result.passed


def test_ctf_evaluate_dag(paths):
    """Test that evaluate_dag summarises the test outcomes and their confidence intervals."""
    framework = CausalTestingFramework(paths)
    framework.setup(**paths)
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


def test_ctf_evaluate_dag_inestimable(scarf_df):
    """Test evaluate_dag on test cases generated from a DAG with no edges."""
    framework = CausalTestingFramework()
    framework.df = scarf_df.query("length_in > 60")
    framework.dag = CausalDAG(datatypes=framework.df.dtypes)
    framework.dag.add_nodes_from(framework.df.columns)
    framework.test_cases = framework.dag.generate_causal_tests()

    results = framework.evaluate_dag()
    expected = pd.Series(
        {
            "FAIL": 1,
            "FAIL_ci_high": 1,
            "FAIL_ci_low": 0,
            "INESTIMABLE": 0,
            "INESTIMABLE_ci_high": 0,
            "INESTIMABLE_ci_low": 0,
            "PASS": 11,
            "PASS_ci_high": 11,
            "PASS_ci_low": 7,
        }
    ).sort_index()
    pd.testing.assert_series_equal(results, expected)
