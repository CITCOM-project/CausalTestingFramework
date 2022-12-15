import unittest
import os
import numpy as np
import pandas as pd
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.causal_test_engine import CausalTestSuite
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.testing.estimators import CausalForestEstimator, LinearRegressionEstimator
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_dag import CausalDAG


class TestCausalTestSuite(unittest.TestCase):
    """
    
    """

    def setUp(self) -> None:
        self.test_suite = CausalTestSuite()
        A = Input("A", float)
        self.A = A
        C = Output("C", float)
        self.C = C
        D = Output("D", float)
        self.D = D
        self.base_test_case = BaseTestCase(A, C)
        self.expected_causal_effect = ExactValue(4)
        test_list = [CausalTestCase(self.base_test_case,
                                    self.expected_causal_effect,
                                    0,
                                    1, ),
                     CausalTestCase(self.base_test_case,
                                    self.expected_causal_effect,
                                    0,
                                    2)]
        self.estimators = [LinearRegressionEstimator]
        self.test_suite.add_test_object(base_test_case=self.base_test_case,
                                        causal_test_case_list=test_list,
                                        estimators=self.estimators)

        temp_dir_path = create_temp_dir_if_non_existent()
        dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        f = open(dag_dot_path, "w")
        f.write(dag_dot)
        f.close()
        self.causal_dag = CausalDAG(dag_dot_path)
        self.scenario = Scenario({A, C, D})

        np.random.seed(1)
        df = pd.DataFrame({"D": list(np.random.normal(60, 10, 1000))})  # D = exogenous
        df["A"] = [1 if d > 50 else 0 for d in df["D"]]
        df["C"] = df["D"] + (4 * (df["A"] + 2))  # C = (4*(A+2)) + D
        self.observational_data_csv_path = os.path.join(temp_dir_path, "observational_data.csv")
        df.to_csv(self.observational_data_csv_path, index=False)

    def test_adding_test_object(self):
        test_suite = CausalTestSuite()
        test_list = [CausalTestCase(self.base_test_case,
                                    self.expected_causal_effect,
                                    0,
                                    1)]
        estimators = [LinearRegressionEstimator]
        test_suite.add_test_object(base_test_case=self.base_test_case,
                                   causal_test_case_list=test_list,
                                   estimators=estimators)
        manual_test_object = {
            self.base_test_case: {"tests": test_list, "estimators": estimators, "estimate_type": "ate"}}
        self.assertEqual(test_suite.test_suite, manual_test_object)

    def test_return_single_test_object(self):
        base_test_case = BaseTestCase(self.A, self.D)

        test_list = [CausalTestCase(self.base_test_case,
                                    self.expected_causal_effect,
                                    0,
                                    1)]
        estimators = [LinearRegressionEstimator]
        self.test_suite.add_test_object(base_test_case=base_test_case,
                                        causal_test_case_list=test_list,
                                        estimators=estimators)

        manual_test_case = {"tests": test_list, "estimators": estimators, "estimate_type": "ate"}

        test_case = self.test_suite.get_single_test_object(base_test_case)

        self.assertEqual(test_case, manual_test_case)

    def test_execute_test_suite_single_base_test_case(self):
        """Check that the test suite can return the correct results from dummy data for a single base_test-case"""
        causal_test_engine = self.create_causal_test_engine()

        causal_test_results = causal_test_engine.execute_test_suite(test_suite=self.test_suite)
        causal_test_case = causal_test_results[self.base_test_case]
        self.assertAlmostEqual(causal_test_case['LinearRegressionEstimator'][0].ate, 4, delta=1e-10)

    def test_execute_test_suite_multiple_base_test_cases(self):
        pass

    def test_execute_test_suite_multiple_estimators(self):
        pass

    def create_causal_test_engine(self):

        causal_specification = CausalSpecification(self.scenario, self.causal_dag)

        data_collector = ObservationalDataCollector(self.scenario, self.observational_data_csv_path)
        causal_test_engine = CausalTestEngine(causal_specification, data_collector)
        return causal_test_engine