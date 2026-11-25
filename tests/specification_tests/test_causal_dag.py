"""
This module tests the main functionality of the CausalDAG class, primarily to do with identification.
"""

import os
import re

import networkx as nx
import pytest

from causal_testing.specification.causal_dag import CausalDAG


def write_dot(tmp_path, filename: str, dot: str) -> str:
    """
    Write a DOT graph to a temporary file and return its path as a string (CausalDAG requires a string path).
    """
    path = tmp_path / filename
    path.write_text(dot)
    return str(path)


@pytest.fixture(name="issue_90_dag_path")
def _issue_90_dag_path(tmp_path):
    """
    Path to a DAG in which Z confounds X and is also a cause of the mediator M.
    """
    return write_dot(tmp_path, "dag.dot", "digraph DAG { rankdir=LR; Z -> X; X -> M; M -> Y; Z -> M; }")


@pytest.fixture(name="iv_dag_path")
def _iv_dag_path(tmp_path):
    """
    Path to a DAG in which I is a valid instrument for the effect of X on Y, with U an unobserved confounder.
    """
    return write_dot(tmp_path, "dag.dot", "digraph G { I -> X; X -> Y; U -> X; U -> Y;}")


@pytest.fixture(name="acyclic_dag_path")
def _acyclic_dag_path(tmp_path):
    """
    Path to a simple acyclic DAG.
    """
    return write_dot(tmp_path, "dag.dot", "digraph G { A -> B; B -> C; D -> A; D -> C;}")


@pytest.fixture(name="two_cycle_dcg_path")
def _two_cycle_dcg_path(tmp_path):
    """
    Path to a directed graph containing the cycle A -> B -> A.
    """
    return write_dot(tmp_path, "dcg.dot", "digraph G { A -> B; B -> A; }")


@pytest.fixture(name="cyclic_dag_path")
def _cyclic_dag_path(tmp_path):
    """
    Path to a directed graph containing the cycle A -> B -> C -> A.
    """
    return write_dot(tmp_path, "dag.dot", "digraph G { A -> B; B -> C; D -> A; D -> C; C -> A;}")


@pytest.fixture(name="identification_dag_path")
def _identification_dag_path(tmp_path):
    """
    Path to the DAG used to test the identification algorithms.
    """
    return write_dot(tmp_path, "dag.dot", "digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}")


def test_graphml(issue_90_dag_path):
    """Test that a DAG loaded from a GraphML (.xml) file matches the same DAG loaded from a DOT file."""
    dot_dag = CausalDAG(issue_90_dag_path)
    xml_dag = CausalDAG(os.path.join("tests", "resources", "data", "dag.xml"))
    assert dot_dag.nodes == xml_dag.nodes
    assert dot_dag.edges == xml_dag.edges


def test_invalid_file_extension():
    """Test that creating a CausalDAG from a file that is not .dot or .xml raises an error."""
    with pytest.raises(
        ValueError, match=re.escape("Unsupported file extension test.csv. We only support .dot and .xml files.")
    ):
        CausalDAG("test.csv")


def test_enumerate_minimal_adjustment_sets_issue_90(issue_90_dag_path):
    """
    Test the CausalDAG class for the resolution of Issue 90.
    Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets.
    """
    causal_dag = CausalDAG(issue_90_dag_path)
    xs, ys = ["X"], ["Y"]
    adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
    assert [{"Z"}] == list(adjustment_sets)


def test_valid_iv(iv_dag_path):
    """Test that check_iv_assumptions accepts a valid instrument."""
    causal_dag = CausalDAG(iv_dag_path)
    assert causal_dag.check_iv_assumptions("X", "Y", "I")


def test_unrelated_instrument(iv_dag_path):
    """Test that check_iv_assumptions rejects an instrument that is not a cause of the treatment."""
    causal_dag = CausalDAG(iv_dag_path)
    causal_dag.remove_edge("I", "X")
    with pytest.raises(ValueError, match=re.escape("Instrument I is not associated with treatment X in the DAG")):
        causal_dag.check_iv_assumptions("X", "Y", "I")


def test_direct_cause(iv_dag_path):
    """Test that check_iv_assumptions rejects an instrument that affects the outcome other than through the
    treatment.
    """
    causal_dag = CausalDAG(iv_dag_path)
    causal_dag.add_edge("I", "Y")
    with pytest.raises(
        ValueError, match=re.escape("Instrument I affects the outcome Y other than through the treatment X")
    ):
        causal_dag.check_iv_assumptions("X", "Y", "I")


def test_common_cause(iv_dag_path):
    """Test that check_iv_assumptions rejects an instrument that shares a common cause with the outcome."""
    causal_dag = CausalDAG(iv_dag_path)
    causal_dag.add_edge("U", "I")
    with pytest.raises(ValueError, match=re.escape("Instrument I and outcome Y share common causes")):
        causal_dag.check_iv_assumptions("X", "Y", "I")


def test_valid_causal_dag(acyclic_dag_path):
    """Test whether the Causal DAG is valid."""
    causal_dag = CausalDAG(acyclic_dag_path)
    assert list(causal_dag.nodes) == ["A", "B", "C", "D"]
    assert list(causal_dag.edges) == [("A", "B"), ("B", "C"), ("D", "A"), ("D", "C")]


def test_add_cycle(acyclic_dag_path):
    """Test that adding an edge that introduces a cycle raises an error."""
    causal_dag = CausalDAG(acyclic_dag_path)
    with pytest.raises(nx.HasACycle):
        causal_dag.add_edge("C", "A")


def test_read_cycle(two_cycle_dcg_path):
    """Test that reading a graph that has a cycle raises an error."""
    with pytest.raises(nx.HasACycle):
        CausalDAG(two_cycle_dcg_path)


def test_empty_causal_dag():
    """Test whether an empty dag can be created."""
    causal_dag = CausalDAG()
    assert list(causal_dag.nodes) == []
    assert list(causal_dag.edges) == []


def test_to_dot_string(acyclic_dag_path):
    """Test that to_dot_string produces the expected DOT representation of the DAG."""
    causal_dag = CausalDAG(acyclic_dag_path)
    assert causal_dag.to_dot_string() == """digraph G {\nA -> B;\nB -> C;\nD -> A;\nD -> C;\n}"""


def test_invalid_causal_dag(cyclic_dag_path):
    """Test that reading a graph containing a cycle raises an error."""
    with pytest.raises(nx.HasACycle):
        CausalDAG(cyclic_dag_path)


def test_ignore_cycles(cyclic_dag_path):
    """Test that a cyclic graph can be read and used for identification when ignore_cycles is set."""
    dag = CausalDAG(cyclic_dag_path, ignore_cycles=True)
    assert dag.identification(treatment_variable="B", outcome_variable="C") == {"A"}


def test_direct_effect_adjustment_sets(identification_dag_path):
    """Test that direct_effect_adjustment_sets finds all adjustment sets for a direct effect."""
    causal_dag = CausalDAG(identification_dag_path)
    adjustment_sets = causal_dag.direct_effect_adjustment_sets(["X1"], ["Y"])
    assert list(adjustment_sets) == [{"D1", "Z"}, {"X2", "Z"}]


def test_direct_effect_adjustment_sets_no_adjustment(identification_dag_path):
    """Test that direct_effect_adjustment_sets returns the empty set when no adjustment is needed."""
    causal_dag = CausalDAG(identification_dag_path)
    adjustment_sets = causal_dag.direct_effect_adjustment_sets(["X2"], ["D1"])
    assert list(adjustment_sets) == [set()]


def test_get_indirect_graph(identification_dag_path):
    """Test that get_indirect_graph removes the edge from treatment to outcome and leaves the rest of the graph
    unchanged.
    """
    causal_dag = CausalDAG(identification_dag_path)
    indirect_graph = causal_dag.get_indirect_graph(["D1"], ["Y"])
    original_edges = list(causal_dag.edges)
    original_edges.remove(("D1", "Y"))
    assert list(indirect_graph.edges) == original_edges
    assert indirect_graph.nodes == causal_dag.nodes


def test_proper_backdoor_graph(identification_dag_path):
    """Test whether converting a Causal DAG to a proper back-door graph works correctly."""
    causal_dag = CausalDAG(identification_dag_path)
    proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(["X1", "X2"], ["Y"])
    edges = set(
        [
            ("X1", "X2"),
            ("X2", "V"),
            ("X2", "D2"),
            ("D1", "D2"),
            ("D1", "Y"),
            ("Y", "D3"),
            ("Z", "X2"),
            ("Z", "Y"),
        ]
    )
    assert set(proper_backdoor_graph.edges) == edges


def test_proper_backdoor_graph_invalid_treatment(identification_dag_path):
    """Test that an invalid treatment variable raises an `IndexError`.."""
    causal_dag = CausalDAG(identification_dag_path)
    with pytest.raises(IndexError):
        causal_dag.get_proper_backdoor_graph(["INVALID"], ["Y"])


def test_constructive_backdoor_criterion_should_hold(identification_dag_path):
    """Test whether the constructive criterion holds when it should."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys, zs = ["X1", "X2"], ["Y"], ["Z"]
    proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
    assert causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs)


def test_constructive_backdoor_criterion_should_not_hold_not_d_separator_in_proper_backdoor_graph(
    identification_dag_path,
):
    """Test whether the constructive criterion fails when the adjustment set is not a d-separator."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys, zs = ["X1", "X2"], ["Y"], ["V"]
    proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
    assert not causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs)


def test_constructive_backdoor_criterion_should_not_hold_descendent_of_proper_causal_path(identification_dag_path):
    """Test that the constructive criterion does not hold when the adjustment set Z contains a descendant of a variable
    on a proper causal path between X and Y."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys, zs = ["X1", "X2"], ["Y"], ["D1"]
    proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
    assert not causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs)


def test_is_min_adjustment_for_min_adjustment(identification_dag_path):
    """Test whether the minimum adjustment set is minimal."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys, zs = ["X1", "X2"], ["Y"], {"Z"}
    assert causal_dag.adjustment_set_is_minimal(xs, ys, zs)


def test_is_min_adjustment_for_not_min_adjustment(identification_dag_path):
    """Test whether the minimum adjustment set is not minimal."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys, zs = ["X1", "X2"], ["Y"], {"Z", "V"}
    assert not causal_dag.adjustment_set_is_minimal(xs, ys, zs)


def test_is_min_adjustment_for_invalid_adjustment(identification_dag_path):
    """Test correct identification of an invalid minimum adjustment set."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys, zs = ["X1", "X2"], ["Y"], set()
    with pytest.raises(ValueError, match=re.escape("set() is not a valid adjustment set.")):
        causal_dag.adjustment_set_is_minimal(xs, ys, zs)


def test_get_ancestor_graph_of_causal_dag(identification_dag_path):
    """Test whether get_ancestor_graph converts a CausalDAG to the correct ancestor graph."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys = ["X1", "X2"], ["Y"]
    ancestor_graph = causal_dag.get_ancestor_graph(xs, ys)
    assert list(ancestor_graph.nodes) == ["X1", "X2", "D1", "Y", "Z"]
    assert list(ancestor_graph.edges) == [("X1", "X2"), ("X2", "D1"), ("D1", "Y"), ("Z", "X2"), ("Z", "Y")]


def test_get_ancestor_graph_of_proper_backdoor_graph(identification_dag_path):
    """Test whether get_ancestor_graph converts a CausalDAG to the correct proper back-door graph."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys = ["X1", "X2"], ["Y"]
    proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
    ancestor_graph = proper_backdoor_graph.get_ancestor_graph(xs, ys)
    assert list(ancestor_graph.nodes) == ["X1", "X2", "D1", "Y", "Z"]
    assert list(ancestor_graph.edges) == [("X1", "X2"), ("D1", "Y"), ("Z", "X2"), ("Z", "Y")]


def test_enumerate_minimal_adjustment_sets(identification_dag_path):
    """Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets."""
    causal_dag = CausalDAG(identification_dag_path)
    xs, ys = ["X1", "X2"], ["Y"]
    adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
    assert [{"Z"}] == list(adjustment_sets)


def test_identification_total_effect():
    """Test whether identification works for total effect."""
    causal_dag = CausalDAG()
    causal_dag.add_edges_from([("X", "M"), ("M", "Y")])

    assert set() == causal_dag.identification(treatment_variable="X", outcome_variable="Y", effect_type="total")


def test_identification_invalid_effect():
    """Test that identification raises an error for an effect type other than total or direct."""
    causal_dag = CausalDAG()
    with pytest.raises(ValueError, match=re.escape("Causal effect should be 'total' or 'direct', not 'invalid'")):
        causal_dag.identification(treatment_variable="X", outcome_variable="Y", effect_type="invalid")


def test_enumerate_minimal_adjustment_sets_multiple():
    """Test whether enumerate_minimal_adjustment_sets lists all minimum adjustment sets if multiple are possible."""
    causal_dag = CausalDAG()
    causal_dag.add_edges_from(
        [
            ("X1", "X2"),
            ("X2", "V"),
            ("Z1", "X2"),
            ("Z1", "Z2"),
            ("Z2", "Z3"),
            ("Z3", "Y"),
            ("D1", "Y"),
            ("D1", "D2"),
            ("Y", "D3"),
        ]
    )
    xs, ys = ["X1", "X2"], ["Y"]
    adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
    set_of_adjustment_sets = set(frozenset(min_separator) for min_separator in adjustment_sets)
    assert {frozenset({"Z1"}), frozenset({"Z2"}), frozenset({"Z3"})} == set_of_adjustment_sets


def test_enumerate_minimal_adjustment_sets_two_adjustments():
    """Test whether enumerate_minimal_adjustment_sets lists all possible minimum adjustment sets of arity two."""
    causal_dag = CausalDAG()
    causal_dag.add_edges_from(
        [
            ("X1", "X2"),
            ("X2", "V"),
            ("Z1", "X2"),
            ("Z1", "Z2"),
            ("Z2", "Z3"),
            ("Z3", "Y"),
            ("D1", "Y"),
            ("D1", "D2"),
            ("Y", "D3"),
            ("Z4", "X1"),
            ("Z4", "Y"),
            ("X2", "D1"),
        ]
    )
    xs, ys = ["X1", "X2"], ["Y"]
    adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
    set_of_adjustment_sets = set(frozenset(min_separator) for min_separator in adjustment_sets)
    assert {frozenset({"Z1", "Z4"}), frozenset({"Z2", "Z4"}), frozenset({"Z3", "Z4"})} == set_of_adjustment_sets


def test_dag_with_non_character_nodes():
    """Test identification for a DAG whose nodes are strings of length greater than 1."""
    causal_dag = CausalDAG()
    causal_dag.add_edges_from(
        [
            ("va", "ba"),
            ("ba", "ia"),
            ("ba", "da"),
            ("ba", "ra"),
            ("la", "va"),
            ("la", "aa"),
            ("aa", "ia"),
            ("aa", "da"),
            ("aa", "ra"),
        ]
    )
    xs, ys = ["ba"], ["da"]
    adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
    assert list(adjustment_sets) == [{"aa"}, {"la"}, {"va"}]


def test_impossible_identification():
    """Test that identification raises an error when the only valid adjustment set contains a node in
    nodes_to_ignore.
    """
    causal_dag = CausalDAG()
    causal_dag.add_edges_from([("X", "M"), ("M", "Y"), ("X", "Y")])

    assert causal_dag.identification(treatment_variable="X", outcome_variable="Y") == {"M"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Could not find a suitable adjustment set for the direct effect of X on Y while avoiding nodes in set "
            "{'M'}."
        ),
    ):
        causal_dag.identification(treatment_variable="X", outcome_variable="Y", nodes_to_ignore=["M"])


def test_adjustment_set_nodes_to_ignore():
    """Test whether identification produces different adjustment sets if nodes_to_ignore is set."""
    causal_dag = CausalDAG()
    causal_dag.add_edges_from([("L", "V"), ("V", "X"), ("X", "Y"), ("L", "C"), ("C", "Y")])

    assert causal_dag.identification(treatment_variable="X", outcome_variable="Y") == {"C"}
    assert causal_dag.identification(treatment_variable="X", outcome_variable="Y", nodes_to_ignore={"C"}) == {"L"}
