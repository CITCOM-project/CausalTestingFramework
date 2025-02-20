import os
import unittest
from pathlib import Path
import scipy
import pandas as pd

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.ipcw_estimator import IPCWEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_adequacy import DAGAdequacy
from causal_testing.testing.causal_test_outcome import NoEffect, SomeEffect
from causal_testing.specification.scenario import Scenario
from causal_testing.testing.causal_test_adequacy import DataAdequacy
from causal_testing.specification.variable import Input, Output
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
        inputs = [
            Input("test_input", float, self.example_distribution),
            Input("test_input_no_dist", float, self.example_distribution),
        ]
        outputs = [Output("test_output", float)]
        self.scenario = Scenario(variables=inputs + outputs)

    def test_data_adequacy_numeric(self):
        base_test_case = BaseTestCase(
            Input("test_input", float, self.example_distribution), Output("test_output", float)
        )
        estimator = LinearRegressionEstimator(
            base_test_case=base_test_case, treatment_value=None, control_value=None, adjustment_set={}, df=self.df
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=NoEffect(),
            estimate_type="coefficient",
            estimator=estimator,
        )
        adequacy_metric = DataAdequacy(causal_test_case, estimator)
        adequacy_metric.measure_adequacy()
        self.assertEqual(
            adequacy_metric.to_dict(),
            {"kurtosis": {"test_input": 0.0}, "bootstrap_size": 100, "passing": 100, "successful": 100},
        )

    def test_data_adequacy_categorical(self):
        base_test_case = BaseTestCase(
            Input("test_input_no_dist", float, self.example_distribution), Output("test_output", float)
        )
        estimator = LinearRegressionEstimator(
            base_test_case=base_test_case, treatment_value=None, control_value=None, adjustment_set={}, df=self.df
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=NoEffect(),
            estimate_type="coefficient",
            estimator=estimator,
        )
        adequacy_metric = DataAdequacy(causal_test_case, estimator)
        adequacy_metric.measure_adequacy()
        self.assertEqual(
            adequacy_metric.to_dict(),
            {"kurtosis": {"test_input_no_dist[T.b]": 0.0}, "bootstrap_size": 100, "passing": 100, "successful": 100},
        )

    def test_data_adequacy_group_by(self):
        timesteps_per_intervention = 1
        control_strategy = [[t, "t", 0] for t in range(1, 4, timesteps_per_intervention)]
        treatment_strategy = [[t, "t", 1] for t in range(1, 4, timesteps_per_intervention)]
        outcome = Output("outcome", float)
        fit_bl_switch_formula = "xo_t_do ~ time"
        df = pd.read_csv("tests/resources/data/temporal_data.csv")
        df["ok"] = df["outcome"] == 1
        estimation_model = IPCWEstimator(
            df,
            timesteps_per_intervention,
            control_strategy,
            treatment_strategy,
            outcome,
            "ok",
            fit_bl_switch_formula=fit_bl_switch_formula,
            fit_bltd_switch_formula=fit_bl_switch_formula,
            eligibility=None,
        )
        base_test_case = BaseTestCase(Input("t", float), Output("outcome", float))

        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=SomeEffect(),
            estimate_type="hazard_ratio",
            estimator=estimation_model,
        )
        adequacy_metric = DataAdequacy(causal_test_case, estimation_model, group_by="id")
        adequacy_metric.measure_adequacy()
        adequacy_dict = adequacy_metric.to_dict()
        self.assertEqual(round(adequacy_dict["kurtosis"]["trtrand"], 3), -0.857)
        adequacy_dict.pop("kurtosis")
        self.assertEqual(
            adequacy_dict,
            {"bootstrap_size": 100, "passing": 32, "successful": 100},
        )

    def test_dag_adequacy_dependent(self):
        base_test_case = BaseTestCase(
            treatment_variable="test_input",
            outcome_variable="B",
            effect=None,
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=None,
            estimate_type=None,
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
        base_test_case = BaseTestCase(
            treatment_variable="test_input",
            outcome_variable="C",
            effect=None,
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=None,
            estimate_type=None,
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
        base_test_case = BaseTestCase(
            treatment_variable="C",
            outcome_variable="test_input",
            effect=None,
        )
        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=None,
            estimate_type=None,
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
