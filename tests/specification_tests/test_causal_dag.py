import os
import re
from pathlib import Path

import networkx as nx
import pytest

from causal_testing.specification.causal_dag import CausalDAG, close_separator, list_all_min_sep


class TestCausalDAGIssue90:
    """
    Test the CausalDAG class for the resolution of Issue 90.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; Z -> X; X -> M; M -> Y; Z -> M; }"""
        Path(self.dag_dot_path).write_text(dag_dot)

    def test_graphml(self):
        dot_dag = CausalDAG(self.dag_dot_path)
        xml_dag = CausalDAG(os.path.join("tests", "resources", "data", "dag.xml"))
        assert dot_dag.nodes == xml_dag.nodes
        assert dot_dag.edges == xml_dag.edges

    def test_invalid_file_extension(self):
        with pytest.raises(
            ValueError, match=re.escape("Unsupported file extension test.csv. We only support .dot and .xml files.")
        ):
            CausalDAG("test.csv")

    def test_enumerate_minimal_adjustment_sets(self):
        """Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X"], ["Y"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        assert [{"Z"}] == list(adjustment_sets)


class TestIVAssumptions:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        dag_dot = """digraph G { I -> X; X -> Y; U -> X; U -> Y;}"""
        Path(self.dag_dot_path).write_text(dag_dot)

    def test_valid_iv(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        assert causal_dag.check_iv_assumptions("X", "Y", "I")

    def test_unrelated_instrument(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.remove_edge("I", "X")
        with pytest.raises(ValueError, match=re.escape("Instrument I is not associated with treatment X in the DAG")):
            causal_dag.check_iv_assumptions("X", "Y", "I")

    def test_direct_cause(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.add_edge("I", "Y")
        with pytest.raises(
            ValueError, match=re.escape("Instrument I affects the outcome Y other than through the treatment X")
        ):
            causal_dag.check_iv_assumptions("X", "Y", "I")

    def test_common_cause(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        causal_dag.add_edge("U", "I")
        with pytest.raises(ValueError, match=re.escape("Instrument I and outcome Y share common causes")):
            causal_dag.check_iv_assumptions("X", "Y", "I")


class TestCausalDAG:
    """
    Test the CausalDAG class for creation of Causal Directed Acyclic Graphs (DAGs).

    In particular, confirm whether the Causal DAG class creates valid causal directed acyclic graphs (empty and directed
    graphs without cycles) and refuses to create invalid (cycle-containing) graphs.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        Path(self.dag_dot_path).write_text("digraph G { A -> B; B -> C; D -> A; D -> C;}")
        self.dcg_dot_path = str(tmp_path / "dcg.dot")
        Path(self.dcg_dot_path).write_text("digraph G { A -> B; B -> A; }")

    def test_valid_causal_dag(self):
        """Test whether the Causal DAG is valid."""
        causal_dag = CausalDAG(self.dag_dot_path)
        assert list(causal_dag.nodes) == ["A", "B", "C", "D"] and list(causal_dag.edges) == [
            ("A", "B"),
            ("B", "C"),
            ("D", "A"),
            ("D", "C"),
        ]

    def test_add_cycle(self):
        """Test that adding an edge that introduces a cycle raises an error."""
        causal_dag = CausalDAG(self.dag_dot_path)
        with pytest.raises(nx.HasACycle):
            causal_dag.add_edge("C", "A")

    def test_read_cycle(self):
        """Test that reading a graph that has a cycle raises an error."""
        with pytest.raises(nx.HasACycle):
            CausalDAG(self.dcg_dot_path)

    def test_empty_casual_dag(self):
        """Test whether an empty dag can be created."""
        causal_dag = CausalDAG()
        assert list(causal_dag.nodes) == [] and list(causal_dag.edges) == []

    def test_to_dot_string(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        assert causal_dag.to_dot_string() == """digraph G {\nA -> B;\nB -> C;\nD -> A;\nD -> C;\n}"""


class TestCyclicCausalDAG:
    """
    Test the creation of a cyclic causal graph.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C; C -> A;}"""
        Path(self.dag_dot_path).write_text(dag_dot)

    def test_invalid_causal_dag(self):
        with pytest.raises(nx.HasACycle):
            CausalDAG(self.dag_dot_path)

    def test_ignore_cycles(self):
        dag = CausalDAG(self.dag_dot_path, ignore_cycles=True)
        assert dag.identification(treatment_variable="B", outcome_variable="C") == {"A"}


class TestDAGDirectEffectIdentification:
    """
    Test the Causal DAG identification algorithms and supporting algorithms.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        dag_dot = """digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}"""
        Path(self.dag_dot_path).write_text(dag_dot)

    def test_direct_effect_adjustment_sets(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        adjustment_sets = causal_dag.direct_effect_adjustment_sets(["X1"], ["Y"])
        assert list(adjustment_sets) == [{"D1", "Z"}, {"X2", "Z"}]

    def test_direct_effect_adjustment_sets_no_adjustment(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        adjustment_sets = causal_dag.direct_effect_adjustment_sets(["X2"], ["D1"])
        assert list(adjustment_sets) == [set()]


class TestDAGIdentification:
    """
    Test the Causal DAG identification algorithms and supporting algorithms.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dag_dot_path = str(tmp_path / "dag.dot")
        dag_dot = """digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}"""
        Path(self.dag_dot_path).write_text(dag_dot)

    def test_get_indirect_graph(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        indirect_graph = causal_dag.get_indirect_graph(["D1"], ["Y"])
        original_edges = list(causal_dag.edges)
        original_edges.remove(("D1", "Y"))
        assert list(indirect_graph.edges) == original_edges
        assert indirect_graph.nodes == causal_dag.nodes

    def test_proper_backdoor_graph(self):
        """Test whether converting a Causal DAG to a proper back-door graph works correctly."""
        causal_dag = CausalDAG(self.dag_dot_path)
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
        assert set(proper_backdoor_graph.edges).issubset(edges)

    def test_proper_backdoor_graph_invalid_tratment(self):
        """Test whether converting a Causal DAG to a proper back-door graph works correctly."""
        causal_dag = CausalDAG(self.dag_dot_path)
        with pytest.raises(IndexError):
            causal_dag.get_proper_backdoor_graph(["INVALID"], ["Y"])

    def test_constructive_backdoor_criterion_should_hold(self):
        """Test whether the constructive criterion holds when it should."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["Z"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        assert causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs)

    def test_constructive_backdoor_criterion_should_not_hold_not_d_separator_in_proper_backdoor_graph(
        self,
    ):
        """Test whether the constructive criterion fails when the adjustment set is not a d-separator."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["V"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        assert not causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs)

    def test_constructive_backdoor_criterion_should_not_hold_descendent_of_proper_causal_path(
        self,
    ):
        """Test whether the constructive criterion holds when the adjustment set Z contains a descendent of a variable
        on a proper causal path between X and Y."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["D1"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        assert not causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs)

    def test_is_min_adjustment_for_min_adjustment(self):
        """Test whether is_min_adjustment can correctly test whether the minimum adjustment set is minimal."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], {"Z"}
        assert causal_dag.adjustment_set_is_minimal(xs, ys, zs)

    def test_is_min_adjustment_for_not_min_adjustment(self):
        """Test whether is_min_adjustment can correctly test whether the minimum adjustment set is not minimal."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], {"Z", "V"}
        assert not causal_dag.adjustment_set_is_minimal(xs, ys, zs)

    def test_is_min_adjustment_for_invalid_adjustment(self):
        """Test whether is min_adjustment can correctly identify that the minimum adjustment set is invalid."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], set()
        with pytest.raises(ValueError, match=re.escape("set() is not a valid adjustment set.")):
            causal_dag.adjustment_set_is_minimal(xs, ys, zs)

    def test_get_ancestor_graph_of_causal_dag(self):
        """Test whether get_ancestor_graph converts a CausalDAG to the correct ancestor graph."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X1", "X2"], ["Y"]
        ancestor_graph = causal_dag.get_ancestor_graph(xs, ys)
        assert list(ancestor_graph.nodes) == ["X1", "X2", "D1", "Y", "Z"]
        assert list(ancestor_graph.edges) == [("X1", "X2"), ("X2", "D1"), ("D1", "Y"), ("Z", "X2"), ("Z", "Y")]

    def test_get_ancestor_graph_of_proper_backdoor_graph(self):
        """Test whether get_ancestor_graph converts a CausalDAG to the correct proper back-door graph."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X1", "X2"], ["Y"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        ancestor_graph = proper_backdoor_graph.get_ancestor_graph(xs, ys)
        assert list(ancestor_graph.nodes) == ["X1", "X2", "D1", "Y", "Z"]
        assert list(ancestor_graph.edges) == [("X1", "X2"), ("D1", "Y"), ("Z", "X2"), ("Z", "Y")]

    def test_enumerate_minimal_adjustment_sets(self):
        """Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ["X1", "X2"], ["Y"]
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        assert [{"Z"}] == list(adjustment_sets)

    def test_identification_total_effect(self):
        """Test whether identification works for total effect."""
        causal_dag = CausalDAG()
        causal_dag.add_edges_from([("X", "M"), ("M", "Y")])

        assert set() == causal_dag.identification(treatment_variable="X", outcome_variable="Y", effect_type="total")

    def test_identification_invalid_effect(self):
        causal_dag = CausalDAG()
        with pytest.raises(ValueError, match=re.escape("Causal effect should be 'total' or 'direct', not 'invalid'")):
            causal_dag.identification(treatment_variable="X", outcome_variable="Y", effect_type="invalid")

    def test_enumerate_minimal_adjustment_sets_multiple(self):
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

    def test_enumerate_minimal_adjustment_sets_two_adjustments(self):
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

    def test_dag_with_non_character_nodes(self):
        """Test identification for a DAG whose nodes are not just characters (strings of length greater than 1)."""
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


class TestUndirectedGraphAlgorithms:
    """
    Test the graph algorithms designed for the undirected graph variants of a Causal DAG.
    During the identification process, a Causal DAG is converted into several forms of undirected graph which allow for
    more efficient computation of minimal separators. This suite of tests covers the two main algorithms applied to
    these forms of undirected graphs (ancestor and moral graphs): close_separator and list_all_min_sep."""

    def setup_method(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edges_from([("a", 2), ("a", 3), (2, 4), (3, 5), (3, 4), (4, "b"), (5, "b")])
        self.treatment_node = "a"
        self.outcome_node = "b"
        self.treatment_node_set = {"a"}
        self.outcome_node_set = set(nx.neighbors(self.graph, "b"))
        self.outcome_node_set.add("b")

    def test_close_separator(self):
        """Test whether close_separator correctly identifies the close separator of {2,3} in the undirected graph."""
        result = close_separator(self.graph, self.treatment_node, self.outcome_node, self.treatment_node_set)
        assert {2, 3} == result

    def test_list_all_min_sep(self):
        """Test whether list_all_min_sep finds all minimal separators for the undirected graph relative to a and b."""
        min_separators = list(
            list_all_min_sep(
                self.graph,
                self.treatment_node,
                self.outcome_node,
                self.treatment_node_set,
                self.outcome_node_set,
            )
        )

        # Convert list of sets to set of frozen sets for comparison
        min_separators = set(frozenset(min_separator) for min_separator in min_separators)
        assert {frozenset({2, 3}), frozenset({3, 4}), frozenset({4, 5})} == min_separators

    def test_close_separator_exception(self):
        g = nx.Graph()
        g.add_edges_from([("X", "Y")])

        with pytest.raises(ValueError, match=re.escape("No X-X separator in the graph.")):
            close_separator(
                graph=g,
                treatment_node="X",
                outcome_node="X",
                treatment_node_set={"Y"},
            )


class TestHiddenVariableDAG:
    """
    Test the CausalDAG identification for the exclusion of hidden variables.
    """

    def test_impossible_identification(self):
        """Test whether identification produces different adjustment sets if nodes_to_ignore is set."""
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

    def test_adjustment_set_nodes_to_ignore(self):
        """Test whether identification produces different adjustment sets if nodes_to_ignore is set."""
        causal_dag = CausalDAG()
        causal_dag.add_edges_from([("L", "V"), ("V", "X"), ("X", "Y"), ("L", "C"), ("C", "Y")])

        assert causal_dag.identification(treatment_variable="X", outcome_variable="Y") == {"C"}
        assert causal_dag.identification(treatment_variable="X", outcome_variable="Y", nodes_to_ignore={"C"}) == {"L"}


def time_it(label, func, *args, **kwargs):
    import time

    start = time.time()
    result = func(*args, **kwargs)
    print(f"{label} took {time.time() - start:.6f} seconds")
    return result
