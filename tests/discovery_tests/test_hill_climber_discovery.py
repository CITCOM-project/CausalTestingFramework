"""
This module tests the Hill Climber Discovery algorithm.
"""

import unittest

import pandas as pd

from causal_testing.discovery.hill_climber_discovery import HillClimberDiscovery
from causal_testing.specification.causal_dag import CausalDAG


class TestHillClimber(unittest.TestCase):

    def test_evaluate_fitness(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        dag = CausalDAG()
        dag.add_nodes_from(scarf_df.columns)
        dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])

        hill_climber = HillClimberDiscovery(scarf_df)
        fitness_values, problem_edges = hill_climber.evaluate_fitness(dag)
        expected_fitness_values = (0.8, 0.0, 0.0, -1.0, 0.0, 0.0)
        expected_problem_edges = [
            ("length_in", "completed"),
            ("large_gauge", "completed"),
        ]

        self.assertEqual(fitness_values, expected_fitness_values)
        self.assertEqual(problem_edges, expected_problem_edges)

    def test_discovery_edges(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        hill_climber = HillClimberDiscovery(
            scarf_df,
            include_edges=[("length_in", "completed")],
            exclude_edges=[("color", "length_in")],
            max_iterations=20,
        )
        dag = hill_climber.discover()
        self.assertTrue(
            ("length_in", "completed") in dag.edges, f"Expected ('length_in', 'completed') to be in {dag.edges}"
        )
        self.assertFalse(
            ("color", "completed") in dag.edges, f"Expected ('color', 'completed') NOT to be in {dag.edges}"
        )
