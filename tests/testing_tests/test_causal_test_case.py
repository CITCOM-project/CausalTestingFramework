import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np

from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase


class TestCausalTestCase(unittest.TestCase):
    """Test the CausalTestCase class.

    The base test case is a data class which contains the minimum information
    necessary to perform identification. The CausalTestCase class represents
    a causal test case. We here test the basic getter methods.
    """

    def setUp(self) -> None:
        # 2. Create Scenario and Causal Specification
        A = Input("A", float)
        C = Output("C", float)

        # 3. Create an intervention and causal test case
        self.expected_causal_effect = ExactValue(4)
        self.base_test_case = BaseTestCase(A, C)
        self.causal_test_case = CausalTestCase(
            base_test_case=self.base_test_case,
            expected_causal_effect=self.expected_causal_effect,
            estimator=LinearRegressionEstimator(
                base_test_case=self.base_test_case,
                adjustment_set=set(),
                control_value=0,
                treatment_value=1,
            ),
        )

    def test_str(self):
        print(str(self.causal_test_case))
        self.assertEqual(
            str(self.causal_test_case),
            "Running {'A': 1} instead of {'A': 0} should cause the following changes to {'C'}: ExactValue: 4±0.2.",
        )


class TestCausalTestExecution(unittest.TestCase):
    """
    Test the causal test execution workflow using observational data.
    """

    def setUp(self) -> None:
        # 1. Create Causal DAG
        self.temp_dir_path = tempfile.mkdtemp()
        dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        with open(dag_dot_path, "w") as file:
            file.write(dag_dot)
        self.causal_dag = CausalDAG(dag_dot_path)

        # 2. Create Scenario and Causal Specification
        self.A = Input("A", float)
        self.C = Output("C", float)
        self.D = Output("D", float)
        self.scenario = Scenario({self.A, self.C, self.D})
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.causal_dag)

        # 3. Create a causal test case
        self.expected_causal_effect = ExactValue(4)
        self.base_test_case_A_C = BaseTestCase(self.A, self.C)
        self.base_test_case_D_A = BaseTestCase(self.D, self.A)
        self.causal_test_case = CausalTestCase(
            base_test_case=self.base_test_case_A_C,
            expected_causal_effect=self.expected_causal_effect,
            # control_value=0,
            # treatment_value=1,
        )

        # 4. Create dummy test data and write to csv
        np.random.seed(1)
        self.df = pd.DataFrame({"D": list(np.random.normal(60, 10, 1000))})  # D = exogenous
        self.df["A"] = [1 if d > 50 else 0 for d in self.df["D"]]
        self.df["C"] = self.df["D"] + (4 * (self.df["A"] + 2))  # C = (4*(A+2)) + D
        # self.observational_data_csv_path = os.path.join(self.temp_dir_path, "observational_data.csv")
        # self.df.to_csv(self.observational_data_csv_path, index=False)

        # 5. Create minimal adjustment set
        self.minimal_adjustment_set = self.causal_dag.identification(self.base_test_case_A_C)
        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)

    def test_check_minimum_adjustment_set(self):
        """Check that the minimum adjustment set is correctly made"""
        minimal_adjustment_set = self.causal_dag.identification(self.base_test_case_A_C)
        self.assertEqual(minimal_adjustment_set, {"D"})

    def test_invalid_causal_effect(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        base_test_case = BaseTestCase(treatment_variable=self.A, outcome_variable=self.C, effect="error")

        with self.assertRaises(Exception):
            self.causal_dag.identification(base_test_case)

    def test_execute_test_observational_linear_regression_estimator(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            self.base_test_case_A_C,
            self.treatment_value,
            self.control_value,
            self.minimal_adjustment_set,
            self.df,
        )
        causal_test_result = self.causal_test_case.execute_test(estimation_model)
        pd.testing.assert_series_equal(causal_test_result.test_value.value, pd.Series(4.0), atol=1e-10)

    def test_execute_test_observational_linear_regression_estimator_direct_effect(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        base_test_case = BaseTestCase(treatment_variable=self.A, outcome_variable=self.C, effect="direct")
        estimation_model = LinearRegressionEstimator(
            self.base_test_case_A_C,
            self.treatment_value,
            self.control_value,
            self.causal_dag.identification(base_test_case),
            self.df,
        )

        causal_test_case = CausalTestCase(
            base_test_case=base_test_case,
            expected_causal_effect=self.expected_causal_effect,
            estimator=estimation_model,
        )

        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0
        causal_test_result = causal_test_case.execute_test()
        pd.testing.assert_series_equal(causal_test_result.test_value.value, pd.Series(4.0), atol=1e-10)

    def test_execute_test_observational_linear_regression_estimator_coefficient(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            self.base_test_case_D_A,
            self.treatment_value,
            self.control_value,
            self.minimal_adjustment_set,
            self.df,
        )
        self.causal_test_case.estimate_type = "coefficient"
        causal_test_result = self.causal_test_case.execute_test(estimation_model)
        pd.testing.assert_series_equal(causal_test_result.test_value.value, pd.Series({"D": 0.0}), atol=1e-1)

    def test_execute_test_observational_linear_regression_estimator_risk_ratio(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            self.base_test_case_D_A,
            self.treatment_value,
            self.control_value,
            self.minimal_adjustment_set,
            self.df,
        )
        self.causal_test_case.estimate_type = "risk_ratio"
        causal_test_result = self.causal_test_case.execute_test(estimation_model)
        pd.testing.assert_series_equal(causal_test_result.test_value.value, pd.Series(0.0), atol=1)

    def test_invalid_estimate_type(self):
        """Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator."""
        estimation_model = LinearRegressionEstimator(
            self.base_test_case_D_A,
            self.treatment_value,
            self.control_value,
            self.minimal_adjustment_set,
            self.df,
        )
        self.causal_test_case.estimate_type = "invalid"
        with self.assertRaises(AttributeError):
            self.causal_test_case.execute_test(estimation_model)

    def test_execute_test_observational_linear_regression_estimator_squared_term(self):
        """Check that executing the causal test case returns the correct results for dummy data with a squared term
        using a linear regression estimator. C ~ 4*(A+2) + D + D^2"""
        estimation_model = LinearRegressionEstimator(
            self.base_test_case_A_C,
            self.treatment_value,
            self.control_value,
            self.minimal_adjustment_set,
            self.df,
            formula=f"C ~ A + {'+'.join(self.minimal_adjustment_set)} + (D ** 2)",
        )
        causal_test_result = self.causal_test_case.execute_test(estimation_model)
        pd.testing.assert_series_equal(causal_test_result.test_value.value, pd.Series(4.0), atol=1)
