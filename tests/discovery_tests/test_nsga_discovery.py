"""
This module tests the NSGA Discovery algorithm.
"""

import unittest
import numpy as np
import networkx as nx
import pandas as pd
from causal_testing.discovery.nsga_discovery import NSGADiscovery
from causal_testing.specification.causal_dag import CausalDAG


class TestNSGA(unittest.TestCase):

    def test_binary_string_to_causal_dag(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        dag = CausalDAG()
        dag.add_nodes_from(scarf_df.columns)
        dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])
        nsga = NSGADiscovery(scarf_df)

        self.assertTrue(
            nx.utils.graphs_equal(dag, nsga.binary_string_to_causal_dag(nsga.causal_dag_to_binary_string(dag)))
        )

    def test_multiobjective_fitness(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        dag = CausalDAG()
        dag.add_nodes_from(scarf_df.columns)
        dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])
        nsga = NSGADiscovery(scarf_df)

        expected = np.array([2, 2, 2, 0.0, 0.0, 2])
        multiobjective_fitness = nsga.multi_objective_fitness(None, nsga.causal_dag_to_binary_string(dag), None)
        self.assertTrue(
            np.array_equal(multiobjective_fitness, expected), f"Arrays differ: {expected} != {multiobjective_fitness}"
        )

    def test_discovery_edges(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        hill_climber = NSGADiscovery(
            scarf_df,
            include_edges=[("length_in", "completed")],
            exclude_edges=[("color", "length_in")],
            max_iterations=10,
        )
        dag = hill_climber.discover()
        self.assertTrue(
            ("length_in", "completed") in dag.edges, f"Expected ('length_in', 'completed') to be in {dag.edges}"
        )
        self.assertFalse(
            ("color", "completed") in dag.edges, f"Expected ('color', 'completed') NOT to be in {dag.edges}"
        )
