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
def _df():
    """
    Data with numeric and categorical inputs.
    """
    return pd.read_csv("tests/resources/data/data_with_categorical.csv")


@pytest.fixture(name="dag")
def _dag():
    """
    Causal DAG corresponding to the data with numeric and categorical inputs.
    """
    return CausalDAG("tests/resources/data/dag.dot")


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


def test_data_adequacy_categorical_inestimable():
    """Test data adequacy when the categorical effect cannot be estimated at all."""
    df = pd.read_csv("tests/resources/data/scarf_data.csv")
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=LinearRegressionEstimator(
            treatment_variable="color", outcome_variable="completed", adjustment_set=set()
        ),
    )
    adequacy_metric = causal_test_case.measure_adequacy(df.loc[df["color"] == "grey"])

    assert adequacy_metric.kurtosis is None, f"Expected passing None not {adequacy_metric.kurtosis}"
    assert adequacy_metric.passing == 0, f"Expected passing 0 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 0, f"Expected successful 0 not {adequacy_metric.successful}"
    assert adequacy_metric.results == []


def test_data_adequacy_categorical_partly_inestimable():
    """Test data adequacy when the categorical effect can only sometimes be estimated."""
    df = pd.read_csv("tests/resources/data/scarf_data.csv")
    causal_test_case = CausalTestCase(
        expected_causal_effect=NoEffect(atol=1e-10),
        effect_measure="coefficient",
        estimator=LinearRegressionEstimator(
            treatment_variable="color", outcome_variable="completed", adjustment_set=set()
        ),
    )
    adequacy_metric = causal_test_case.measure_adequacy(df.loc[df["length_in"] == 55])

    assert adequacy_metric.kurtosis.values == [0], f"Expected [0] not {adequacy_metric.kurtosis.values}"
    assert adequacy_metric.passing == 63, f"Expected passing 63 not {adequacy_metric.passing}"
    assert adequacy_metric.successful == 63, f"Expected successful 63 not {adequacy_metric.successful}"


def test_data_adequacy_group_by():
    """Test data adequacy for an IPCW estimator with bootstrap samples grouped by individual."""
    timesteps_per_intervention = 1
    control_strategy = [[t, "t", 0] for t in range(1, 4, timesteps_per_intervention)]
    treatment_strategy = [[t, "t", 1] for t in range(1, 4, timesteps_per_intervention)]
    fit_bl_switch_formula = "xo_t_do ~ time"
    df = pd.read_csv("tests/resources/data/temporal_data.csv")
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
    adequacy_metric = causal_test_case.measure_adequacy(df, bootstrap_size=10)

    expected_dict = {
        "kurtosis": {"test_input": 0.0},
        "passing": 10,
        "successful": 10,
        "bootstrap_size": 10,
        "results": {
            "effect_estimate": {
                0: -2.220446049250313e-16,
                1: -1.1102230246251565e-16,
                2: 7.632783294297951e-17,
                3: 5.551115123125783e-17,
                4: 6.938893903907228e-17,
                5: 5.551115123125783e-17,
                6: 4.163336342344337e-17,
                7: -1.6653345369377348e-16,
                8: 3.469446951953614e-17,
                9: -1.3877787807814457e-16,
            },
            "ci_low": {
                0: -7.771553129157294e-16,
                1: -2.279269223868238e-16,
                2: -1.8906584687236155e-16,
                3: -1.57411847966069e-16,
                4: -3.27696708597398e-17,
                5: 2.3327440223904727e-17,
                6: -6.643195454332691e-19,
                7: -3.158788196765981e-16,
                8: -1.878128765814627e-16,
                9: -3.348328858540416e-16,
            },
            "ci_high": {
                0: 3.3306610306566683e-16,
                1: 5.8823174617924694e-18,
                2: 3.4172151275832057e-16,
                3: 2.6843415042858463e-16,
                4: 1.7154754893788438e-16,
                5: 8.769486223861093e-17,
                6: 8.393104639232001e-17,
                7: -1.718808771094884e-17,
                8: 2.57201815620535e-16,
                9: 5.727712969775247e-17,
            },
            "test_index": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
            "passed": {
                0: True,
                1: True,
                2: True,
                3: True,
                4: True,
                5: True,
                6: True,
                7: True,
                8: True,
                9: True,
            },
            "var": {
                0: "test_input",
                1: "test_input",
                2: "test_input",
                3: "test_input",
                4: "test_input",
                5: "test_input",
                6: "test_input",
                7: "test_input",
                8: "test_input",
                9: "test_input",
            },
        },
    }
    # Use json_normalize to avoid rounding errors
    pd.testing.assert_frame_equal(
        pd.json_normalize(expected_dict).round(2),
        pd.json_normalize(adequacy_metric.to_dict(include_adequacy_results=True)).round(2),
    )


def test_dag_adequacy_dependent(dag):
    """Test DAG adequacy when the tested pair is causally dependent."""
    causal_test_case = CausalTestCase(
        estimator=LinearRegressionEstimator(
            treatment_variable="test_input", outcome_variable="B", adjustment_set=set()
        ),
        expected_causal_effect=None,
        effect_measure=None,
    )
    test_suite = [causal_test_case]
    dag_adequacy = DAGAdequacy(dag, test_suite)
    dag_adequacy.measure_adequacy()
    assert dag_adequacy.to_dict() == {
        "causal_dag": dag,
        "test_suite": test_suite,
        "tested_pairs": {("test_input", "B")},
        "pairs_to_test": {
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
        },
        "untested_pairs": {
            ("B", "C"),
            ("test_input_no_dist", "test_input"),
            ("C", "test_output"),
            ("test_input_no_dist", "B"),
            ("test_input", "test_output"),
            ("test_input", "C"),
            ("test_input_no_dist", "test_output"),
            ("B", "test_output"),
            ("test_input_no_dist", "C"),
        },
        "dag_adequacy": 0.1,
    }


def test_dag_adequacy_independent(dag):
    """Test DAG adequacy when the tested pair is causally independent."""
    causal_test_case = CausalTestCase(
        estimator=LinearRegressionEstimator(
            treatment_variable="test_input", outcome_variable="C", adjustment_set=set()
        ),
        expected_causal_effect=None,
        effect_measure=None,
    )
    test_suite = [causal_test_case]
    dag_adequacy = DAGAdequacy(dag, test_suite)
    dag_adequacy.measure_adequacy()
    assert dag_adequacy.to_dict() == {
        "causal_dag": dag,
        "test_suite": test_suite,
        "tested_pairs": {("test_input", "C")},
        "pairs_to_test": {
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
        },
        "untested_pairs": {
            ("B", "C"),
            ("test_input_no_dist", "test_input"),
            ("C", "test_output"),
            ("test_input_no_dist", "B"),
            ("test_input", "test_output"),
            ("test_input", "B"),
            ("test_input_no_dist", "test_output"),
            ("B", "test_output"),
            ("test_input_no_dist", "C"),
        },
        "dag_adequacy": 0.1,
    }


def test_dag_adequacy_independent_other_way(dag):
    """Test DAG adequacy when the tested pair is causally independent and specified the other way around."""
    causal_test_case = CausalTestCase(
        estimator=LinearRegressionEstimator(
            treatment_variable="C", outcome_variable="test_input", adjustment_set=set()
        ),
        expected_causal_effect=None,
        effect_measure=None,
    )
    test_suite = [causal_test_case]
    dag_adequacy = DAGAdequacy(dag, test_suite)
    dag_adequacy.measure_adequacy()
    assert dag_adequacy.to_dict() == {
        "causal_dag": dag,
        "test_suite": test_suite,
        "tested_pairs": {("test_input", "C")},
        "pairs_to_test": {
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
        },
        "untested_pairs": {
            ("B", "C"),
            ("test_input_no_dist", "test_input"),
            ("C", "test_output"),
            ("test_input_no_dist", "B"),
            ("test_input", "test_output"),
            ("test_input", "B"),
            ("test_input_no_dist", "test_output"),
            ("B", "test_output"),
            ("test_input_no_dist", "C"),
        },
        "dag_adequacy": 0.1,
    }
