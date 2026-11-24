"""
This module tests common causal discovery functionality provided within the abstract_discovery module.
"""

import networkx as nx
import pandas as pd
import pytest

from causal_testing.discovery.abstract_discovery import Discovery, simple_cycle
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome


class AbstractDiscovery(Discovery):
    """
    Minimal concrete implementation of abstract "Discovery" class.
    """

    def discover(self):
        """
        Minimal implementation of the abstract discover method.
        """
        return CausalDAG()


def test_simple_cycle():
    """
    Test that simple_cycle returns the edges of a cycle in the graph.
    """
    dag = CausalDAG(ignore_cycles=True)
    dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    assert simple_cycle(dag) == [("A", "B"), ("B", "C"), ("C", "A")]


def test_simple_cycle_no_cycles():
    """
    Test that simple_cycle returns an empty list for an acyclic graph.
    """
    dag = CausalDAG()
    dag.add_edges_from([("A", "B"), ("B", "C")])
    assert simple_cycle(dag) == []


def test_include_edge_wildcard():
    """
    Test that regex wildcards in include_edges are expanded to all matching variables.
    """
    abstract_discovery = AbstractDiscovery(
        df=pd.DataFrame(columns=["x_1", "x_2", "x_3", "y_1", "y_2", "y_3", "z_1", "z_2"]),
        include_edges=[("x_.*", "y_1")],
    )
    assert abstract_discovery.include_edges == [(f"x_{n}", "y_1") for n in range(1, 4)]


def test_include_edge_cycle():
    """
    Test that specifying include_edges that form a cycle raises a HasACycle error.
    """
    with pytest.raises(nx.exception.HasACycle):
        AbstractDiscovery(
            df=pd.DataFrame(columns=["x_1", "x_2", "x_3", "y_1", "y_2", "y_3", "z_1", "z_2"]),
            include_edges=[("x_1", "y_1"), ("y_1", "x_1")],
        )


def test_exclude_edge_wildcard():
    """
    Test that regex wildcards in exclude_edges are expanded to all matching variables.
    """
    abstract_discovery = AbstractDiscovery(
        df=pd.DataFrame(columns=["x_1", "x_2", "x_3", "y_1", "y_2", "y_3", "z_1", "z_2"]),
        exclude_edges=[("x_.*", "y_1")],
    )
    assert abstract_discovery.exclude_edges == [(f"x_{n}", "y_1") for n in range(1, 4)]


def test_remove_cycles():
    """
    Test that remove_cycles makes a cyclic graph acyclic.
    """
    dag = CausalDAG(ignore_cycles=True)
    dag.add_edges_from([("A", "B"), ("B", "C")])
    dag.add_edge("C", "A")
    assert not dag.is_acyclic(), "A -> B -> C -> A should form a cycle."

    abstract_discovery = AbstractDiscovery(pd.DataFrame())
    abstract_discovery.remove_cycles(dag)
    assert dag.is_acyclic()


def test_remove_cycles_respects_include_edges():
    """
    Test that remove_cycles does not remove edges that are in include_edges.
    """
    dag = CausalDAG(ignore_cycles=True)
    dag.add_edges_from([("A", "B"), ("B", "C")])
    dag.add_edge("C", "A")

    include_edges = {("A", "B"), ("B", "C")}
    abstract_discovery = AbstractDiscovery(pd.DataFrame(columns=dag.nodes), include_edges=include_edges)

    abstract_discovery.remove_cycles(dag)
    assert dag.is_acyclic()
    for edge in include_edges:
        assert edge in dag.edges, f"{edge} not in {dag.edges}"


def test_remove_cycles_no_cycles_present():
    """
    Test that remove_cycles leaves an acyclic graph unchanged.
    """
    dag = CausalDAG()
    dag.add_edges_from([("A", "B")])

    abstract_discovery = AbstractDiscovery(pd.DataFrame())
    abstract_discovery.remove_cycles(dag)
    assert len(dag.edges()) == 1


def test_remove_cycles_multiple_cycles():
    """
    Test that remove_cycles removes exactly one edge from each of multiple independent cycles.
    """
    dag = CausalDAG(ignore_cycles=True)
    dag.add_edges_from([("A", "B"), ("C", "D"), ("B", "A"), ("D", "C")])

    abstract_discovery = AbstractDiscovery(pd.DataFrame())
    abstract_discovery.remove_cycles(dag)
    assert len(dag.edges()) == 2
    assert dag.has_edge("A", "B") or dag.has_edge("B", "A")
    assert dag.has_edge("C", "D") or dag.has_edge("D", "C")


def test_evaluate_tests_invalid_datatype(scarf_df):
    """
    Test that evaluate_tests raises a ValueError when the data contains an unsupported datatype.
    """
    scarf_df["completed"] = pd.to_datetime(["2026-01-01" for _ in range(len(scarf_df))], format="%Y-%m-%d")

    dag = CausalDAG()
    dag.add_nodes_from(scarf_df.columns)
    dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])

    abstract_discovery = AbstractDiscovery(scarf_df)
    with pytest.raises(ValueError):
        abstract_discovery.evaluate_tests(dag)


@pytest.mark.parametrize(
    ("filter_data", "completed_color_outcome"),
    [
        pytest.param(False, TestOutcome.PASS, id="full_data"),
        pytest.param(True, TestOutcome.INESTIMABLE, id="inestimable"),
    ],
)
def test_evaluate_tests(scarf_df, filter_data, completed_color_outcome):
    """
    Test that evaluate_tests produces the expected test outcomes for the scarf example, including
    inestimable outcomes when the data is insufficient to estimate an effect.
    """
    if filter_data:
        # Keep only completed, non-grey scarves, so the effect of completed on color cannot be estimated
        scarf_df["completed"] = scarf_df["completed"].astype(bool)
        scarf_df = scarf_df.loc[(scarf_df["completed"]) & (scarf_df["color"] != "grey")]

    dag = CausalDAG()
    dag.add_nodes_from(scarf_df.columns)
    dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])

    abstract_discovery = AbstractDiscovery(scarf_df)
    test_results = abstract_discovery.evaluate_tests(dag)
    expected_results = pd.DataFrame(
        [
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "length_in",
                "outcome": "large_gauge",
            },
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "large_gauge",
                "outcome": "length_in",
            },
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "length_in",
                "outcome": "color",
            },
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "color",
                "outcome": "length_in",
            },
            {
                "result": TestOutcome.FAIL,
                "expected_effect": "SomeEffect",
                "treatment": "length_in",
                "outcome": "completed",
            },
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "large_gauge",
                "outcome": "color",
            },
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "color",
                "outcome": "large_gauge",
            },
            {
                "result": TestOutcome.FAIL,
                "expected_effect": "SomeEffect",
                "treatment": "large_gauge",
                "outcome": "completed",
            },
            {
                "result": TestOutcome.PASS,
                "expected_effect": "NoEffect",
                "treatment": "color",
                "outcome": "completed",
            },
            {
                "result": completed_color_outcome,
                "expected_effect": "NoEffect",
                "treatment": "completed",
                "outcome": "color",
            },
        ]
    )
    pd.testing.assert_frame_equal(test_results, expected_results)
