"""
This module tests the NSGA Discovery algorithm.
"""

import numpy as np

from causal_testing.discovery.nsga_discovery import NSGADiscovery
from causal_testing.specification.causal_dag import CausalDAG


def test_binary_string_to_causal_dag(scarf_df):
    dag = CausalDAG()
    dag.add_nodes_from(scarf_df.columns)
    dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])
    nsga = NSGADiscovery(scarf_df)

    back_translated_dag = nsga.binary_string_to_causal_dag(nsga.causal_dag_to_binary_string(dag))

    assert dag.nodes == back_translated_dag.nodes
    assert dag.edges == back_translated_dag.edges


def test_multiobjective_fitness(scarf_df):
    dag = CausalDAG()
    dag.add_nodes_from(scarf_df.columns)
    dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])
    nsga = NSGADiscovery(scarf_df)

    expected = np.array([2, 2, 2, 0.0, 0.0, 2])
    multiobjective_fitness = nsga.multi_objective_fitness(None, nsga.causal_dag_to_binary_string(dag), None)
    np.testing.assert_array_equal(multiobjective_fitness, expected)


def test_discovery_edges(scarf_df):
    nsga = NSGADiscovery(
        scarf_df,
        include_edges=[("length_in", "completed")],
        exclude_edges=[("color", "length_in")],
        max_iterations=20,
    )
    dag = nsga.discover()
    assert ("length_in", "completed") in dag.edges, f"Expected ('length_in', 'completed') to be in {dag.edges}"
    assert ("color", "completed") not in dag.edges, f"Expected ('color', 'completed') NOT to be in {dag.edges}"
