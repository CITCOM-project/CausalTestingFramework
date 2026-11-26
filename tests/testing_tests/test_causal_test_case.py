"""
Test the causal test execution workflow using observational data.
"""

import re

import numpy as np
import pandas as pd
import pytest

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_effect import ExactValue
from causal_testing.testing.causal_test_case import CausalTestCase

TREATMENT_VALUE = 1
CONTROL_VALUE = 0


@pytest.fixture(name="causal_dag")
def _causal_dag():
    """
    Causal DAG in which D confounds the effect of A on C.
    """
    causal_dag = CausalDAG()
    causal_dag.add_edges_from([("A", "C"), ("D", "A"), ("D", "C")])
    return causal_dag


@pytest.fixture(name="expected_causal_effect")
def _expected_causal_effect():
    """
    The expected causal effect of A on C.
    """
    return ExactValue(4)


@pytest.fixture(name="minimal_adjustment_set")
def _minimal_adjustment_set(causal_dag):
    """
    The minimal adjustment set for the effect of A on C.
    """
    return causal_dag.identification(treatment_variable="A", outcome_variable="C")


@pytest.fixture(name="estimator")
def _estimator(minimal_adjustment_set):
    """
    Linear regression estimator for the effect of A on C.
    """
    return LinearRegressionEstimator(
        treatment_variable="A",
        outcome_variable="C",
        treatment_value=TREATMENT_VALUE,
        control_value=CONTROL_VALUE,
        adjustment_set=minimal_adjustment_set,
    )


@pytest.fixture(name="causal_test_case")
def _causal_test_case(estimator, expected_causal_effect):
    """
    Causal test case for the ATE of A on C.
    """
    return CausalTestCase(
        estimator=estimator,
        expected_causal_effect=expected_causal_effect,
        effect_measure="ate",
    )


@pytest.fixture(name="df")
def _df():
    """
    Dummy observational data where C = (4*(A+2)) + D.
    """
    np.random.seed(1)
    df = pd.DataFrame({"D": list(np.random.normal(60, 10, 1000))})  # D = exogenous
    df["A"] = [1 if d > 50 else 0 for d in df["D"]]
    df["C"] = df["D"] + (4 * (df["A"] + 2))  # C = (4*(A+2)) + D
    return df


def test_treatment_no_estimator(expected_causal_effect):
    """Check that a test case with no estimator has no treatment variable."""
    assert (
        CausalTestCase(expected_causal_effect=expected_causal_effect, effect_measure="ate").treatment_variable is None
    )


def test_outcome_no_estimator(expected_causal_effect):
    """Check that a test case with no estimator has no outcome variable."""
    assert CausalTestCase(expected_causal_effect=expected_causal_effect, effect_measure="ate").outcome_variable is None


def test_check_minimum_adjustment_set(minimal_adjustment_set):
    """Check that the minimum adjustment set is correctly made"""
    assert minimal_adjustment_set == {"D"}


def test_execute_test_observational_linear_regression_estimator_direct_effect(causal_test_case, df):
    """Check that executing the causal test case returns the correct results for dummy data using a linear
    regression estimator."""
    effect_estimate = causal_test_case.estimate_effect(df)
    pd.testing.assert_series_equal(effect_estimate.effect_estimate, pd.Series(4.0), atol=1e-10)


def test_execute_test_observational_linear_regression_estimator_coefficient(causal_dag, df):
    """Check that executing the causal test case returns the correct results for dummy data using a linear
    regression estimator."""
    causal_test_case = CausalTestCase(
        expected_causal_effect=None,
        estimator=LinearRegressionEstimator(
            treatment_variable="D",
            outcome_variable="A",
            treatment_value=TREATMENT_VALUE,
            control_value=CONTROL_VALUE,
            adjustment_set=causal_dag.identification(treatment_variable="D", outcome_variable="A"),
        ),
        effect_measure="coefficient",
    )
    effect_estimate = causal_test_case.estimate_effect(df)
    pd.testing.assert_series_equal(effect_estimate.effect_estimate, pd.Series({"D": 0.0}), atol=1e-1)


def test_execute_test_observational_linear_regression_estimator_risk_ratio(causal_dag, df):
    """Check that executing the causal test case returns the correct results for dummy data using a linear
    regression estimator."""
    causal_test_case = CausalTestCase(
        expected_causal_effect=None,
        estimator=LinearRegressionEstimator(
            treatment_variable="D",
            outcome_variable="A",
            treatment_value=TREATMENT_VALUE,
            control_value=CONTROL_VALUE,
            adjustment_set=causal_dag.identification(treatment_variable="D", outcome_variable="A"),
        ),
        effect_measure="risk_ratio",
    )
    effect_estimate = causal_test_case.estimate_effect(df)
    pd.testing.assert_series_equal(effect_estimate.effect_estimate, pd.Series(0.0), atol=1)


def test_invalid_effect_measure(causal_dag, df):
    """Test invalid causal effect measure raises an error."""
    causal_test_case = CausalTestCase(
        expected_causal_effect=None,
        estimator=LinearRegressionEstimator(
            treatment_variable="D",
            outcome_variable="A",
            treatment_value=TREATMENT_VALUE,
            control_value=CONTROL_VALUE,
            adjustment_set=causal_dag.identification(treatment_variable="D", outcome_variable="A"),
        ),
        effect_measure="invalid",
    )
    with pytest.raises(AttributeError, match=re.escape(f"{LinearRegressionEstimator} has no invalid method.")):
        causal_test_case.execute_test(df)


def test_execute_test_observational_linear_regression_estimator_squared_term(
    expected_causal_effect, minimal_adjustment_set, df
):
    """Check that executing the causal test case returns the correct results for dummy data with a squared term
    using a linear regression estimator. C ~ 4*(A+2) + D + D^2"""
    causal_test_case = CausalTestCase(
        expected_causal_effect=expected_causal_effect,
        estimator=LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            treatment_value=TREATMENT_VALUE,
            control_value=CONTROL_VALUE,
            adjustment_set=minimal_adjustment_set,
            formula=f"C ~ A + {'+'.join(minimal_adjustment_set)} + (D ** 2)",
        ),
        effect_measure="ate",
    )
    effect_estimate = causal_test_case.estimate_effect(df)
    pd.testing.assert_series_equal(effect_estimate.effect_estimate, pd.Series(4.0), atol=1)


def test_estimate_params_with_formula(expected_causal_effect, df):
    """Ensure estimate params is handled correctly when a formula is passed into the estimator object"""
    causal_test_case = CausalTestCase(
        expected_causal_effect=expected_causal_effect,
        effect_measure="risk_ratio",
        estimator=LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            control_value=0,
            treatment_value=1,
            formula="C ~ A + D",
            adjustment_config={"D": 1},
        ),
    )
    assert round(causal_test_case.estimate_effect(df).effect_estimate[0], 3) == 1.444


def test_to_dict(df):
    """Check that a causal test case and its result convert to a dict."""
    causal_test_case = CausalTestCase(
        name="A |- C",
        expected_causal_effect=ExactValue(4),
        effect_measure="coefficient",
        estimator=LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            formula="C ~ A + D",
        ),
    )
    causal_test_case.execute_test(df, adequacy=True)
    test_case_dict = causal_test_case.to_dict()

    expected = {
        "name": "A |- C",
        "skip": False,
        "effect_measure": "coefficient",
        "query": None,
        "expected_causal_effect": {"name": "ExactValue", "effect_type": "direct", "value": 4, "atol": 0},
        "estimator": {
            "name": "LinearRegressionEstimator",
            "treatment_variable": "A",
            "outcome_variable": "C",
            "alpha": 0.05,
            "adjustment_set": ["D"],
            "formula": "C ~ A + D",
        },
        "result": {
            "outcome": "PASS",
            "passed": True,
            "effect_estimate": {
                "effect_measure": "coefficient",
                "effect_estimate": {"A": 4.0},
                "ci_low": {"A": 4.0},
                "ci_high": {"A": 4.0},
            },
            "adequacy": {
                "kurtosis": {"A": 0.0},
                "kurtosis_simplified": {"A": 100.0},
                "passing": 100,
                "successful": 100,
                "bootstrap_size": 100,
            },
            "effect_direction": "Positive",
        },
    }

    # Use json_normalize to avoid rounding errors
    pd.testing.assert_frame_equal(
        pd.json_normalize(expected).round(2),
        pd.json_normalize(test_case_dict).round(2),
    )
