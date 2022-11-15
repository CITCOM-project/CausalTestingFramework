import unittest
import os

import pandas as pd

from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import Scenario
from causal_testing.specification.metamorphic_relation import ShouldCause
from causal_testing.data_collection.data_collector import ExperimentalDataCollector
from causal_testing.specification.variable import Input, Output


def program_under_test(X1, X2, X3, Z=None, M=None, Y=None):
    if Z is None:
        Z = 2*X1 + -3*X2 + 10
    if M is None:
        M = 3*Z + X3
    if Y is None:
        Y = M/2
    return {'Z': Z, 'M': M, 'Y': Y}


def buggy_program_under_test(X1, X2, X3, Z=None, M=None, Y=None):
    if Z is None:
        Z = 2  # No effect of X1 or X2 on Z
    if M is None:
        M = 3*Z + X3
    if Y is None:
        Y = M/2
    return {'Z': Z, 'M': M, 'Y': Y}


class ProgramUnderTestEDC(ExperimentalDataCollector):

    def run_system_with_input_configuration(self, input_configuration: dict) -> pd.DataFrame:
        results_dict = program_under_test(**input_configuration)
        results_df = pd.DataFrame(results_dict, index=[0])
        return results_df


class BuggyProgramUnderTestEDC(ExperimentalDataCollector):

    def run_system_with_input_configuration(self, input_configuration: dict) -> pd.DataFrame:
        results_dict = buggy_program_under_test(**input_configuration)
        results_df = pd.DataFrame(results_dict, index=[0])
        return results_df


class TestMetamorphicRelation(unittest.TestCase):

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; X1 -> Z; Z -> M; M -> Y; X1 -> M; X2 -> Z; X3 -> M;}"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

        X1 = Input('X1', float)
        X2 = Input('X2', float)
        X3 = Input('X3', float)
        Z = Output('Z', float)
        M = Output('M', float)
        Y = Output('Y', float)
        self.scenario = Scenario(variables={X1, X2, X3, Z, M, Y})
        self.default_control_input_config = {'X1': 1, 'X2': 2, 'X3': 3}
        self.default_treatment_input_config = {'X1': 2, 'X2': 3, 'X3': 3}
        self.data_collector = ProgramUnderTestEDC(self.scenario,
                                                  self.default_control_input_config,
                                                  self.default_treatment_input_config)

    def test_should_cause_metamorphic_relations_should_pass(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        for edge in causal_dag.graph.edges:
            (u, v) = edge
            should_cause_MR = ShouldCause(u, v, None, causal_dag)
            should_cause_MR.generate_follow_up(10, -10.0, 10.0, 1)
            test_results = should_cause_MR.execute_tests(self.data_collector)
            should_cause_MR.test_oracle(test_results)

    def test_should_cause_metamorphic_relation_missing_relationship(self):
        """Test whether the ShouldCause MR catches missing relationships in the DAG."""
        causal_dag = CausalDAG(self.dag_dot_path)
        self.data_collector = BuggyProgramUnderTestEDC(self.scenario,
                                                       self.default_control_input_config,
                                                       self.default_treatment_input_config)
        X1_should_cause_Z_MR = ShouldCause('X1', 'Z', None, causal_dag)
        X2_should_cause_Z_MR = ShouldCause('X2', 'Z', None, causal_dag)
        X1_should_cause_Z_MR.generate_follow_up(10, -100, 100, 1)
        X2_should_cause_Z_MR.generate_follow_up(10, -100, 100, 1)
        X1_should_cause_Z_test_results = X1_should_cause_Z_MR.execute_tests(self.data_collector)
        X2_should_cause_Z_test_results = X2_should_cause_Z_MR.execute_tests(self.data_collector)
        self.assertRaises(AssertionError, X1_should_cause_Z_MR.test_oracle, X1_should_cause_Z_test_results)
        self.assertRaises(AssertionError, X2_should_cause_Z_MR.test_oracle, X2_should_cause_Z_test_results)


