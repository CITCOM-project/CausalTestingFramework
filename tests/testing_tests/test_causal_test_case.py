import unittest
import os
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue
from causal_testing.testing.base_test_case import BaseTestCase


class TestCausalTestEngineObservational(unittest.TestCase):
    """Test the CausalTestEngine workflow using observational data.

    The causal test engine (CTE) is the main workflow for the causal testing framework. The CTE takes a causal test case
    and a causal specification and computes the causal effect of the intervention on the outcome of interest.
    """

    def setUp(self) -> None:
        # 1. Create Causal DAG
        temp_dir_path = create_temp_dir_if_non_existent()
        dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph G { A -> C; D -> A; D -> C}"""
        f = open(dag_dot_path, "w")
        f.write(dag_dot)
        f.close()
        self.causal_dag = CausalDAG(dag_dot_path)

        # 2. Create Scenario and Causal Specification
        A = Input("A", float)
        C = Output("C", float)
        D = Output("D", float)
        self.scenario = Scenario({A, C, D})
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.causal_dag)

        # 3. Create an intervention and causal test case
        self.expected_causal_effect = ExactValue(4)
        self.base_test_case = BaseTestCase(A, C)
        self.causal_test_case = CausalTestCase(
            base_test_case=self.base_test_case,
            expected_causal_effect=self.expected_causal_effect,
            control_value=0,
            treatment_value=1,
        )

    def test_get_treatment_variable(self):
        self.assertEqual(self.causal_test_case.get_treatment_variable(), "A")

    def test_get_outcome_variable(self):
        self.assertEqual(self.causal_test_case.get_outcome_variable(), "C")

    def test_get_treatment_value(self):
        self.assertEqual(self.causal_test_case.get_treatment_value(), 1)

    def test_get_control_value(self):
        self.assertEqual(self.causal_test_case.get_control_value(), 0)

    def test_str(self):
        self.assertEqual(
            str(self.causal_test_case),
            "Running {'A': 1} instead of {'A': 0} should cause the following changes to"
            " {Output: C::float}: ExactValue: 4±0.2.",
        )

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
