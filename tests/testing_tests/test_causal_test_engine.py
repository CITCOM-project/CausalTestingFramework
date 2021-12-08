import unittest
import os
import pandas as pd
import numpy as np
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import CausalForestEstimator, LinearRegressionEstimator


class TestCausalTestEngineObservational(unittest.TestCase):
    """ Test the CausalTestEngine workflow using observational data.

    The causal test engine (CTE) is the main workflow for the causal testing framework. The CTE takes a causal test case
    and a causal specification and computes the causal effect of the intervention on the outcome of interest.
    """

    def setUp(self) -> None:
        # 1. Create Causal DAG
        temp_dir_path = create_temp_dir_if_non_existent()
        dag_dot_path = os.path.join(temp_dir_path, 'dag.dot')
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        f = open(dag_dot_path, 'w')
        f.write(dag_dot)
        f.close()
        self.causal_dag = CausalDAG(dag_dot_path)

        # 2. Create Scenario and Causal Specification
        self.scenario = Scenario()  # TODO: (@MF replace this with updated Scenario/Constraints)
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.causal_dag)

        # 3. Create an intervention and causal test case
        self.intervention = Intervention(('A',), (1,))
        self.expected_causal_effect = {'C': 4}
        self.causal_test_case = CausalTestCase({'A': 0}, self.intervention, self.expected_causal_effect)

        # 4. Create causal test engine
        self.causal_test_engine = CausalTestEngine(self.causal_test_case, self.causal_specification)

        # 5. Create dummy test data and write to csv
        np.random.seed(1)
        df = pd.DataFrame({'D': list(np.random.normal(60, 10, 1000))})  # D = exogenous
        print(df['D'].min(), df['D'].max())
        df['A'] = [1 if d > 50 else 0 for d in df['D']]
        df['C'] = df['D'] + (4 * (df['A'] + 2))  # C = (4*(A+2)) + D
        self.observational_data_csv_path = os.path.join(temp_dir_path, 'observational_data.csv')
        df.to_csv(self.observational_data_csv_path, index=False)

        # 6. Easier to access treatment and outcome values
        self.treatment_value = 1
        self.control_value = 0

    def test_check_no_positivity_violation(self):
        """ Check that no positivity violation is identified when there is no positivity violation. """
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        minimal_adjustment_sets = self.causal_dag.enumerate_minimal_adjustment_sets(['A'], ['C'])
        minimum_adjustment_set = min(minimal_adjustment_sets, key=len)
        variables_to_check = list(minimum_adjustment_set) + ['A'] + ['C']
        self.assertFalse(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_check_positivity_violation_missing_confounder(self):
        """ Check that a positivity violation is identified when there is a positivity violation due to a missing
        confounder. """
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        self.causal_test_engine.scenario_execution_data_df.drop(columns=['D'], inplace=True)  # Remove confounder
        minimal_adjustment_sets = self.causal_dag.enumerate_minimal_adjustment_sets(['A'], ['C'])
        minimum_adjustment_set = min(minimal_adjustment_sets, key=len)
        variables_to_check = list(minimum_adjustment_set) + ['A'] + ['C']
        self.assertTrue(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_check_positivity_violation_missing_treatment(self):
        """ Check that a positivity violation is identified when there is a positivity violation due to a missing
        treatment. """
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        self.causal_test_engine.scenario_execution_data_df.drop(columns=['A'], inplace=True)  # Remove confounder
        minimal_adjustment_sets = self.causal_dag.enumerate_minimal_adjustment_sets(['A'], ['C'])
        minimum_adjustment_set = min(minimal_adjustment_sets, key=len)
        variables_to_check = list(minimum_adjustment_set) + ['A'] + ['C']
        self.assertTrue(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_check_positivity_violation_missing_outcome(self):
        """ Check that a positivity violation is identified when there is a positivity violation due to a missing
        outcome. """
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        self.causal_test_engine.scenario_execution_data_df.drop(columns=['C'], inplace=True)  # Remove confounder
        minimal_adjustment_sets = self.causal_dag.enumerate_minimal_adjustment_sets(['A'], ['C'])
        minimum_adjustment_set = min(minimal_adjustment_sets, key=len)
        variables_to_check = list(minimum_adjustment_set) + ['A'] + ['C']
        self.assertTrue(self.causal_test_engine._check_positivity_violation(variables_to_check))

    def test_execute_test_observational_causal_forest_estimator(self):
        """ Check that executing the causal test case returns the correct results for the dummy data using a causal
        forest estimator. """
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        estimation_model = CausalForestEstimator(('A',),
                                                 self.treatment_value,
                                                 self.control_value,
                                                 {'D'},
                                                 ('C',),
                                                 self.causal_test_engine.scenario_execution_data_df)
        causal_test_result = self.causal_test_engine.execute_test(estimation_model)
        print(causal_test_result)
        self.assertAlmostEqual(causal_test_result.ate, 4, delta=1)

    def test_execute_test_observational_linear_regression_estimator(self):
        """ Check that executing the causal test case returns the correct results for dummy data using a linear
        regression estimator. """
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        estimation_model = LinearRegressionEstimator(('A',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     {'D'},
                                                     ('C',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        causal_test_result = self.causal_test_engine.execute_test(estimation_model)
        self.assertEqual(int(causal_test_result.ate), 4)

    def test_execute_test_observational_linear_regression_estimator_squared_term(self):
        """ Check that executing the causal test case returns the correct results for dummy data with a squared term
        using a linear regression estimator. C ~ 4*(A+8) + D + D^2"""
        self.causal_test_engine.load_data(self.observational_data_csv_path)
        estimation_model = LinearRegressionEstimator(('A',),
                                                     self.treatment_value,
                                                     self.control_value,
                                                     {'D'},
                                                     ('C',),
                                                     self.causal_test_engine.scenario_execution_data_df)
        estimation_model.add_squared_term_to_df('D')
        causal_test_result = self.causal_test_engine.execute_test(estimation_model)
        self.assertAlmostEqual(round(causal_test_result.ate, 1), 4, delta=1)

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
