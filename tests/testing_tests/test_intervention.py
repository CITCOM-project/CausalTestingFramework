import unittest
import os
import pandas as pd
import numpy as np
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_specification import CausalSpecification, Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.causal_test_outcome import ExactValue
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
        self.A = Input("A", float)
        self.scenario = Scenario({self.A})
        self.causal_specification = CausalSpecification(scenario=self.scenario, causal_dag=self.causal_dag)

        # 3. Create an intervention and causal test case
        self.intervention = Intervention((self.A,), (1,))

    def test_apply(self):
        self.assertEqual(self.intervention.apply({self.A: 0}), {self.A: 1})


    def test_str(self):
        self.assertEqual(str(self.intervention), "{A -> 1}")


    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
