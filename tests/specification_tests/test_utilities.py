"""
This module tests the graph algorithms designed for the undirected graph variants of a Causal DAG.
During the identification process, a Causal DAG is converted into several forms of undirected graph which allow for
more efficient computation of minimal separators. This suite of tests covers the two main algorithms applied to
these forms of undirected graphs (ancestor and moral graphs): close_separator and list_all_min_sep."""

import re

import networkx as nx
import pytest

from causal_testing.specification.causal_dag import close_separator, list_all_min_sep


@pytest.fixture(name="undirected_graph")
def _undirected_graph():
    """
    Undirected graph with treatment node "a" and outcome node "b", used to test the minimal separator algorithms.
    """
    graph = nx.Graph()
    graph.add_edges_from([("a", 2), ("a", 3), (2, 4), (3, 5), (3, 4), (4, "b"), (5, "b")])
    return graph


def test_close_separator(undirected_graph):
    """Test whether close_separator correctly identifies the close separator of {2,3} in the undirected graph."""
    result = close_separator(undirected_graph, "a", "b", {"a"})
    assert {2, 3} == result


def test_list_all_min_sep(undirected_graph):
    """Test whether list_all_min_sep finds all minimal separators for the undirected graph relative to a and b."""
    min_separators = list(
        list_all_min_sep(
            undirected_graph,
            "a",
            "b",
            {"a"},
            set(nx.neighbors(undirected_graph, "b")) | {"b"},
        )
    )

    # Convert list of sets to set of frozen sets for comparison
    min_separators = set(frozenset(min_separator) for min_separator in min_separators)
    assert {frozenset({2, 3}), frozenset({3, 4}), frozenset({4, 5})} == min_separators


def test_close_separator_exception():
    """Test that close_separator raises an error when there is no separator between the treatment and outcome."""
    g = nx.Graph()
    g.add_edges_from([("X", "Y")])

    with pytest.raises(ValueError, match=re.escape("No X-X separator in the graph.")):
        close_separator(
            graph=g,
            treatment_node="X",
            outcome_node="X",
            treatment_node_set={"Y"},
        )
