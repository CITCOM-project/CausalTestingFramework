"""
Test the causal test adequacy metrics. These provide metrics determine how comprehensive a test set is and,
ultimately whether we can stop testing.
"""

import pandas as pd
import pytest

from causal_testing.estimation.ipcw_estimator import IPCWEstimator
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_effect import NoEffect, SomeEffect
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.dag_adequacy import DAGAdequacy


@pytest.fixture(name="df")
def _df(resources_dir):
    """
    Data with numeric and categorical inputs.
    """
    return pd.read_csv(resources_dir / "data_with_categorical.csv")


@pytest.fixture(name="dag")
def _dag(resources_dir):
    """
    Causal DAG corresponding to the data with numeric and categorical inputs.
    """
    return CausalDAG(resources_dir / "dag.dot")


def test_data_adequacy_numeric(df):
    """Test data adequacy for a numeric treatment variable."""
    estimator = LinearRegressionEstimator(
        treatment_variable="test_input", outcome_variable="test_output", adjustment_set=set()
    )
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=estimator,
    )
    adequacy_metric = causal_test_case.measure_adequacy(df)

    # This adds a numerical tolerance for Pandas
    assert adequacy_metric.kurtosis["test_input"] == pytest.approx(
        0, abs=1.0
    ), f"Expected kurtosis near 0, got {adequacy_metric.kurtosis['test_input']}"
    assert adequacy_metric.passing == 100, f"Expected passing 100 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 100, f"Expected successful 100 not {adequacy_metric.successful}"


def test_data_adequacy_categorical(df):
    """Test data adequacy for a categorical treatment variable."""
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=LinearRegressionEstimator(
            treatment_variable="test_input_no_dist", outcome_variable="test_output", adjustment_set=set()
        ),
    )
    adequacy_metric = causal_test_case.measure_adequacy(df)

    assert adequacy_metric.kurtosis["test_input_no_dist[T.b]"] == pytest.approx(
        0, abs=1.0
    ), f"Expected kurtosis near 0, got {adequacy_metric.kurtosis['test_input_no_dist[T.b]']}"
    assert adequacy_metric.passing == 100, f"Expected passing 100 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 100, f"Expected successful 100 not {adequacy_metric.successful}"


def test_data_adequacy_categorical_inestimable(scarf_df):
    """Test data adequacy when the categorical effect cannot be estimated at all."""
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=LinearRegressionEstimator(
            treatment_variable="color", outcome_variable="completed", adjustment_set=set()
        ),
    )
    adequacy_metric = causal_test_case.measure_adequacy(scarf_df.loc[scarf_df["color"] == "grey"])

    assert adequacy_metric.kurtosis is None, f"Expected passing None not {adequacy_metric.kurtosis}"
    assert adequacy_metric.passing == 0, f"Expected passing 0 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 0, f"Expected successful 0 not {adequacy_metric.successful}"
    assert adequacy_metric.results == []


def test_data_adequacy_categorical_partly_inestimable(scarf_df):
    """Test data adequacy when the categorical effect can only sometimes be estimated."""
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=LinearRegressionEstimator(
            treatment_variable="color", outcome_variable="completed", adjustment_set=set()
        ),
    )
    adequacy_metric = causal_test_case.measure_adequacy(scarf_df.loc[scarf_df["length_in"] == 55])

    assert adequacy_metric.kurtosis.to_dict() == {
        "color[T.orange]": 0
    }, f"Expected {{'color[T.orange]': 0}} not {adequacy_metric.kurtosis.to_dict()}"
    assert adequacy_metric.passing == 63, f"Expected passing 63 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 63, f"Expected successful 63 not {adequacy_metric.successful}"


def test_data_adequacy_group_by(resources_dir):
    """Test data adequacy for an IPCW estimator with bootstrap samples grouped by individual."""
    timesteps_per_intervention = 1
    control_strategy = [[t, "t", 0] for t in range(1, 4, timesteps_per_intervention)]
    treatment_strategy = [[t, "t", 1] for t in range(1, 4, timesteps_per_intervention)]
    fit_bl_switch_formula = "xo_t_do ~ time"
    df = pd.read_csv(resources_dir / "temporal_data.csv")
    df["ok"] = df["outcome"] == 1
    estimation_model = IPCWEstimator(
        timesteps_per_observation=timesteps_per_intervention,
        control_strategy=control_strategy,
        treatment_strategy=treatment_strategy,
        outcome_variable="outcome",
        status_column="ok",
        fit_bl_switch_formula=fit_bl_switch_formula,
        fit_bltd_switch_formula=fit_bl_switch_formula,
        eligibility=None,
    )

    causal_test_case = CausalTestCase(
        expected_causal_effect=SomeEffect(),
        effect_measure="hazard_ratio",
        estimator=estimation_model,
    )
    adequacy_metric = causal_test_case.measure_adequacy(df, group_by="id")

    assert (
        round(adequacy_metric.kurtosis["trtrand"], 3) == -0.857
    ), f"Expected kurtosis not {round(adequacy_metric.kurtosis['trtrand'], 3)}"
    assert adequacy_metric.passing == 32, f"Expected passing 32 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 100, f"Expected successful 100 not {adequacy_metric.successful}"


def test_to_dict(df):
    """Test that a data adequacy metric converts to a dict, including the adequacy results."""
    estimator = LinearRegressionEstimator(
        treatment_variable="test_input", outcome_variable="test_output", adjustment_set=set()
    )
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=estimator,
    )
    bootstrap_size = 10
    adequacy_metric = causal_test_case.measure_adequacy(df, bootstrap_size=bootstrap_size)

    expected_dict = {
        "kurtosis": {"test_input": 0.0},
        "passing": bootstrap_size,
        "successful": bootstrap_size,
        "bootstrap_size": bootstrap_size,
        "results": {
            "effect_estimate": {i: 0.0 for i in range(bootstrap_size)},
            "ci_low": {i: 0.0 for i in range(bootstrap_size)},
            "ci_high": {i: 0.0 for i in range(bootstrap_size)},
            "test_index": {i: i for i in range(bootstrap_size)},
            "passed": {i: True for i in range(bootstrap_size)},
            "var": {i: "test_input" for i in range(bootstrap_size)},
        },
    }
    # Use json_normalize to avoid rounding errors
    pd.testing.assert_frame_equal(
        pd.json_normalize(expected_dict).round(2),
        pd.json_normalize(adequacy_metric.to_dict(include_adequacy_results=True)).round(2),
    )


@pytest.mark.parametrize(
    ["treatment", "outcome", "tested_pair"],
    [
        pytest.param("test_input", "B", ("test_input", "B"), id="dependent"),
        pytest.param("test_input", "C", ("test_input", "C"), id="independent"),
        pytest.param("C", "test_input", ("test_input", "C"), id="independent_other_way"),
    ],
)
def test_dag_adequacy(dag, treatment, outcome, tested_pair):
    """
    Test DAG adequacy for a test suite containing a single test case. Exactly one pair should be tested, regardless
    of whether the treatment and outcome are causally dependent or which way round they are specified.
    """
    causal_test_case = CausalTestCase(
        estimator=LinearRegressionEstimator(
            treatment_variable=treatment, outcome_variable=outcome, adjustment_set=set()
        ),
        expected_causal_effect=None,
        effect_measure=None,
    )
    test_suite = [causal_test_case]
    dag_adequacy = DAGAdequacy(dag, test_suite)
    dag_adequacy.measure_adequacy()

    dag_pairs_to_test = {
        ("B", "C"),
        ("test_input_no_dist", "test_input"),
        ("C", "test_output"),
        ("test_input", "B"),
        ("test_input_no_dist", "B"),
        ("test_input", "test_output"),
        ("test_input", "C"),
        ("test_input_no_dist", "test_output"),
        ("B", "test_output"),
        ("test_input_no_dist", "C"),
    }

    assert dag_adequacy.to_dict() == {
        "causal_dag": dag,
        "test_suite": test_suite,
        "tested_pairs": {tested_pair},
        "pairs_to_test": dag_pairs_to_test,
        "untested_pairs": dag_pairs_to_test - {tested_pair},
        "dag_adequacy": 0.1,
    }
