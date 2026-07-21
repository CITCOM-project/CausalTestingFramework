import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_effect import ExactValue
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator


class TestCausalTestCase(unittest.TestCase):
    """
    Test the causal test execution workflow using observational data.
    """

    def setUp(self) -> None:
        # Create Causal DAG
        self.temp_dir_path = tempfile.mkdtemp()
        dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        with open(dag_dot_path, "w") as file:
            file.write(dag_dot)
        self.causal_dag = CausalDAG(dag_dot_path)

        # Create a causal test case
        self.expected_causal_effect = ExactValue(4)
        self.causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            effect_measure="ate",
        )

        # Create dummy test data
        np.random.seed(1)
        self.df = pd.DataFrame({"D": list(np.random.normal(60, 10, 1000))})  # D = exogenous
        self.df["A"] = [1 if d > 50 else 0 for d in self.df["D"]]
        self.df["C"] = self.df["D"] + (4 * (self.df["A"] + 2))  # C = (4*(A+2)) + D

        # Create minimal adjustment set
        self.minimal_adjustment_set = self.causal_dag.identification(treatment_variable="A", outcome_variable="C")
        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)

    def test_check_minimum_adjustment_set(self):
        """Check that the minimum adjustment set is correctly made"""
        minimal_adjustment_set = self.causal_dag.identification(treatment_variable="A", outcome_variable="C")
        self.assertEqual(minimal_adjustment_set, {"D"})

    def test_execute_test_observational_linear_regression_estimator(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            treatment_value=self.treatment_value,
            control_value=self.control_value,
            adjustment_set=self.minimal_adjustment_set,
        )
        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
            effect_measure="ate",
        )
        effect_estimate = causal_test_case.estimate_effect(self.df)
        pd.testing.assert_series_equal(effect_estimate.value, pd.Series(4.0), atol=1e-10)

    def test_execute_test_observational_linear_regression_estimator_direct_effect(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            treatment_value=self.treatment_value,
            control_value=self.control_value,
            adjustment_set=self.causal_dag.identification(treatment_variable="A", outcome_variable="C"),
        )

        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
            effect_measure="ate",
        )

        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0
        effect_estimate = causal_test_case.estimate_effect(self.df)
        pd.testing.assert_series_equal(effect_estimate.value, pd.Series(4.0), atol=1e-10)

    def test_execute_test_observational_linear_regression_estimator_coefficient(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            treatment_variable="D",
            outcome_variable="A",
            treatment_value=self.treatment_value,
            control_value=self.control_value,
            adjustment_set=self.minimal_adjustment_set,
        )
        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
            effect_measure="coefficient",
        )
        effect_estimate = causal_test_case.estimate_effect(self.df)
        pd.testing.assert_series_equal(effect_estimate.value, pd.Series({"D": 0.0}), atol=1e-1)

    def test_execute_test_observational_linear_regression_estimator_risk_ratio(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            treatment_variable="D",
            outcome_variable="A",
            treatment_value=self.treatment_value,
            control_value=self.control_value,
            adjustment_set=self.minimal_adjustment_set,
        )
        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
            effect_measure="risk_ratio",
        )
        effect_estimate = causal_test_case.estimate_effect(self.df)
        pd.testing.assert_series_equal(effect_estimate.value, pd.Series(0.0), atol=1)

    def test_invalid_effect_measure(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            treatment_variable="D",
            outcome_variable="A",
            treatment_value=self.treatment_value,
            control_value=self.control_value,
            adjustment_set=self.minimal_adjustment_set,
        )
        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
            effect_measure="invalid",
        )
        with self.assertRaises(AttributeError):
            causal_test_case.execute_test(self.df)

    def test_execute_test_observational_linear_regression_estimator_squared_term(self):
        """Check that executing the causal test case returns the correct results for dummy data with a squared term
        using a linear regression estimator. C ~ 4*(A+2) + D + D^2"""
        estimation_model = LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            treatment_value=self.treatment_value,
            control_value=self.control_value,
            adjustment_set=self.minimal_adjustment_set,
            formula=f"C ~ A + {'+'.join(self.minimal_adjustment_set)} + (D ** 2)",
        )
        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
            effect_measure="ate",
        )
        effect_estimate = causal_test_case.estimate_effect(self.df)
        pd.testing.assert_series_equal(effect_estimate.value, pd.Series(4.0), atol=1)

    def test_estimate_params_with_formula(self):
        """Ensure estimate params is handled correctly when a formula is passed into the estimator object"""

        estimator = LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            adjustment_set=set(),
            control_value=0,
            treatment_value=1,
            formula="C ~ A + D",
            adjustment_config={"D": 1},
        )
        causal_test_case = CausalTestCase(
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=self.expected_causal_effect,
            effect_measure="risk_ratio",
            estimator=estimator,
        )
        self.assertEqual(
            round(
                causal_test_case.estimate_effect(self.df).value[0],
                3,
            ),
            1.444,
        )

    def test_to_dict(self):
        estimator = LinearRegressionEstimator(
            treatment_variable="A",
            outcome_variable="C",
            adjustment_set=set(),
            formula="C ~ A + D",
        )
        causal_test_case = CausalTestCase(
            name="A |- C",
            treatment_variable="A",
            outcome_variable="C",
            expected_causal_effect=ExactValue(4),
            effect_measure="coefficient",
            estimator=estimator,
        )
        causal_test_case.execute_test(self.df, adequacy=True)

        expected = {
            "name": "A |- C",
            "treatment_variable": "A",
            "outcome_variable": "C",
            "skip": False,
            "effect_measure": "coefficient",
            "query": None,
            "expected_effect": {"name": "ExactValue", "effect_type": "direct", "value": 4, "atol": 0.2},
            "estimator": {
                "name": "LinearRegressionEstimator",
                "alpha": 0.05,
                "adjustment_set": [],
                "formula": "C ~ A + D",
            },
            "result": {
                "outcome": "PASS",
                "passed": True,
                "effect_measure": "coefficient",
                "effect_estimate": {"A": 4.0},
                "ci_low": {"A": 4.0},
                "ci_high": {"A": 4.0},
                "adequacy": {"kurtosis": {"A": 0.0}, "passing": 100, "successful": 100},
            },
        }

        # Use json_normalize to avoid rounding errors
        pd.testing.assert_frame_equal(
            pd.json_normalize(expected).round(2), pd.json_normalize(causal_test_case.to_dict()).round(2)
        )
