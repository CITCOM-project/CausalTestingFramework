import os
import unittest
from itertools import cycle
from tempfile import TemporaryDirectory

import pandas as pd

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import CausalTestResult, TestOutcome
from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter, green, red


class TestVisualiser(unittest.TestCase):
    def setUp(self) -> None:
        dag = CausalDAG()
        dag.add_edges_from(
            [
                ("width", "num_lines_abs"),
                ("width", "num_shapes_abs"),
                ("width", "num_lines_unit"),
                ("width", "num_shapes_unit"),
                ("height", "num_lines_abs"),
                ("height", "num_shapes_abs"),
                ("height", "num_lines_unit"),
                ("height", "num_shapes_unit"),
                ("num_lines_abs", "num_lines_unit"),
                ("num_shapes_abs", "num_shapes_unit"),
                ("intensity", "num_lines_abs"),
                ("num_lines_abs", "num_shapes_ab"),
            ]
        )
        self.dag = dag
        ctf = CausalTestingFramework(dag=dag)
        ctf.load_test_cases_from_json("tests/resources/data/poisson_line_tests.json")
        self.plotter = VisualisationPlotter(ctf)

    def test_results_dag_nodes(self):
        """
        The result DAG should have the same nodes as the original.
        """
        assert set(self.plotter.results_dag().nodes) == set(self.dag.nodes)

    def test_results_dag_causal_edges(self):
        """
        The result DAG should have the same causal edges as the original, plus dashed edges for failing independence
        tests. Passing edges should be green. Failing edges should be red.
        """
        results_dag = self.plotter.results_dag()
        for test in self.plotter.ctf.test_cases:
            treatment_variable = test.estimator.treatment_variable
            outcome_variable = test.estimator.outcome_variable
            edge_data = results_dag.get_edge_data(treatment_variable, outcome_variable)
            if not test.result.passed:
                assert edge_data.get("color") == red
            elif (treatment_variable, outcome_variable) in self.plotter.ctf.dag:
                assert edge_data.get("color") == green
