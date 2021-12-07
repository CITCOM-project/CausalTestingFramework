import unittest
import os

import pandas as pd
import numpy as np
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_case import CausalTestCase, CausalTestResult
from causal_testing.testing.causal_test_engine import CausalTestEngine


class TestCausalTestEngine(unittest.TestCase):
    """ Test the CausalTestEngine workflow.

    The causal test engine (CTE) is the main workflow for the causal testing framework. The CTE takes a causal test case
    and a causal specification and computes the causal effect of the intervention on the outcome of interest.
    """

    def setUp(self) -> None:
        # 1. Create Causal DAG
        temp_dir_path = create_temp_dir_if_non_existent()
        dag_dot_path = os.path.join(temp_dir_path, 'dag.dot')
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C}"""
        f = open(dag_dot_path, 'w')
        f.write(dag_dot)
        f.close()
        self.causal_dag = CausalDAG(dag_dot_path)

        # 2. Create Scenario and Causal Specification
        self.scenario = Scenario()  # TODO: (@MF replace this with updated Scenario/Constraints)
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.causal_dag)

        # 3. Create an intervention and causal test case
        self.intervention = Intervention(('A',), (10,))
        self.expected_causal_effect = {'B': 10}
        self.causal_test_case = CausalTestCase({'A': 5}, self.intervention, self.expected_causal_effect)

        # 4. Create causal test engine
        self.causal_test_engine = CausalTestEngine(self.causal_test_case, self.causal_specification)

        # 5. Create dummy test data and write to csv
        df = pd.DataFrame({'D': list(np.random.randint(0, 100, 1000))})  # D = exogenous
        df['A'] = 2 * df['D']  # A = 2*D
        df['C'] = df['D'] * (4*df['A'])  # C = (4*A)*D
        self.observational_data_csv_path = os.path.join(temp_dir_path, 'observational_data.csv')
        df.to_csv(self.observational_data_csv_path, index=False)

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
