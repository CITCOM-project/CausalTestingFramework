import os
import unittest
import scipy
import pandas as pd

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.ipcw_estimator import IPCWEstimator

from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.dag_adequacy import DAGAdequacy
from causal_testing.testing.causal_effect import NoEffect, SomeEffect

from causal_testing.testing.data_adequacy import DataAdequacy

from causal_testing.specification.causal_dag import CausalDAG


class TestCausalTestAdequacy(unittest.TestCase):
    """
    Test the causal test adequacy metrics. These provide metrics determine how comprehensive a test set is and,
    ultimately whether we can stop testing.
    """

    def setUp(self) -> None:
        self.df = pd.read_csv("tests/resources/data/data_with_categorical.csv")
        self.dag = CausalDAG("tests/resources/data/dag.dot")
        self.example_distribution = scipy.stats.uniform(1, 10)

    def test_data_adequacy_numeric(self):
        estimator = LinearRegressionEstimator(
            treatment_variable="test_input",
            outcome_variable="test_output",
        )
        causal_test_case = CausalTestCase(
            treatment_variable="test_input",
            outcome_variable="test_output",
            expected_causal_effect=NoEffect(atol=1e-10),
            effect_measure="coefficient",
            estimator=estimator,
        )
        adequacy_metric = causal_test_case.measure_adequacy(self.df)

        self.assertAlmostEqual(
            adequacy_metric.kurtosis["test_input"],
            0,
            delta=1.0,
            msg=f"Expected kurtosis near 0, got {adequacy_metric.kurtosis['test_input']}",
        )  # This adds a numerical tolerance for Pandas
        self.assertEqual(adequacy_metric.passing, 100, f"Expected passing 100 not {adequacy_metric.passing}")
        self.assertEqual(adequacy_metric.successful, 100, f"Expected successful 100 not {adequacy_metric.successful}")

    def test_data_adequacy_categorical(self):
        causal_test_case = CausalTestCase(
            treatment_variable="test_input_no_dist",
            outcome_variable="test_output",
            expected_causal_effect=NoEffect(atol=1e-10),
            effect_measure="coefficient",
            estimator=LinearRegressionEstimator(
                treatment_variable="test_input_no_dist", outcome_variable="test_output"
            ),
        )
        adequacy_metric = causal_test_case.measure_adequacy(self.df)

        self.assertAlmostEqual(
            adequacy_metric.kurtosis["test_input_no_dist[T.b]"],
            0,
            delta=1.0,
            msg=f"Expected kurtosis near 0, got {adequacy_metric.kurtosis['test_input_no_dist[T.b]']}",
        )
        self.assertEqual(adequacy_metric.passing, 100, f"Expected passing 100 not {adequacy_metric.passing}")
        self.assertEqual(adequacy_metric.successful, 100, f"Expected successful 100 not {adequacy_metric.successful}")

    def test_data_adequacy_group_by(self):
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
            treatment_variable="t",
            outcome_variable="outcome",
            expected_causal_effect=SomeEffect(),
            effect_measure="hazard_ratio",
            estimator=estimation_model,
        )
        adequacy_metric = causal_test_case.measure_adequacy(df, group_by="id")

        self.assertEqual(
            round(adequacy_metric.kurtosis["trtrand"], 3),
            -0.857,
            f"Expected kurtosis not {round(adequacy_metric.kurtosis['trtrand'], 3)}",
        )
        self.assertEqual(adequacy_metric.passing, 32, f"Expected passing 32 not {adequacy_metric.passing}")
        self.assertEqual(adequacy_metric.successful, 100, f"Expected successful 100 not {adequacy_metric.successful}")

    def test_dag_adequacy_dependent(self):
        causal_test_case = CausalTestCase(
            treatment_variable="test_input",
            outcome_variable="B",
            expected_causal_effect=None,
            effect_measure=None,
        )
        test_suite = [causal_test_case]
        dag_adequacy = DAGAdequacy(self.dag, test_suite)
        dag_adequacy.measure_adequacy()
        self.assertEqual(
            dag_adequacy.to_dict(),
            {
                "causal_dag": self.dag,
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
            },
        )

    def test_dag_adequacy_independent(self):
        causal_test_case = CausalTestCase(
            treatment_variable="test_input",
            outcome_variable="C",
            expected_causal_effect=None,
            effect_measure=None,
        )
        test_suite = [causal_test_case]
        dag_adequacy = DAGAdequacy(self.dag, test_suite)
        dag_adequacy.measure_adequacy()
        self.assertEqual(
            dag_adequacy.to_dict(),
            {
                "causal_dag": self.dag,
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
            },
        )

    def test_dag_adequacy_independent_other_way(self):
        causal_test_case = CausalTestCase(
            treatment_variable="C",
            outcome_variable="test_input",
            expected_causal_effect=None,
            effect_measure=None,
        )
        test_suite = [causal_test_case]
        dag_adequacy = DAGAdequacy(self.dag, test_suite)
        dag_adequacy.measure_adequacy()
        self.assertEqual(
            dag_adequacy.to_dict(),
            {
                "causal_dag": self.dag,
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
            },
        )

    def tearDown(self) -> None:
        if os.path.exists("temp_out.txt"):
            os.remove("temp_out.txt")
