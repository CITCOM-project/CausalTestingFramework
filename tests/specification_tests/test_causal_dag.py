import unittest
import os
import networkx as nx
import numpy as np
from causal_testing.specification.causal_dag import CausalDAG, close_separator, list_all_min_sep


class TestCausalDAG(unittest.TestCase):

    def setUp(self) -> None:
        self.dag_dot_path = 'temp/dag.dot'
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C}"""
        f = open(self.dag_dot_path, 'w')
        f.write(dag_dot)
        f.close()

    def test_valid_causal_dag(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        assert list(causal_dag.graph.nodes) == ['A', 'B', 'C', 'D'] and list(causal_dag.graph.edges) == [('A', 'B'),
                                                                                                         ('B', 'C'),
                                                                                                         ('D', 'A'),
                                                                                                         ('D', 'C')]

    def test_invalid_causal_dag(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        self.assertRaises(nx.HasACycle, causal_dag.add_edge, 'C', 'A')

    def test_empty_casual_dag(self):
        causal_dag = CausalDAG()
        assert list(causal_dag.graph.nodes) == [] and list(causal_dag.graph.edges) == []

    def tearDown(self) -> None:
        os.remove(self.dag_dot_path)


class TestDAGIdentification(unittest.TestCase):

    def setUp(self) -> None:
        self.dag_dot_path = 'temp/dag.dot'
        dag_dot = """digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}"""
        f = open(self.dag_dot_path, 'w')
        f.write(dag_dot)
        f.close()

    def test_proper_backdoor_graph(self):
        """
        Test whether converting a Causal DAG to a proper back-door graph works correctly.
        A proper back-door graph should remove the first edge from all proper causal paths from X to Y, where
        X is the set of treatments and Y is the set of outcomes.
        """
        causal_dag = CausalDAG(self.dag_dot_path)
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(['X1', 'X2'], ['Y'])
        self.assertEqual(list(proper_backdoor_graph.graph.edges),
                         [('X1', 'X2'), ('X2', 'V'), ('X2', 'D2'), ('D1', 'D2'), ('D1', 'Y'), ('Y', 'D3'), ('Z', 'X2'),
                          ('Z', 'Y')])

    def test_constructive_backdoor_criterion_should_hold(self):
        """ Test whether the constructive criterion holds when it should. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ['X1', 'X2'], ['Y'], ['Z']
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertTrue(causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs))

    def test_constructive_backdoor_criterion_should_not_hold_not_d_separator_in_proper_backdoor_graph(self):
        """ Test whether the constructive criterion holds when the adjustment set Z is not a d-separator in the proper
        back-door graph. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ['X1', 'X2'], ['Y'], ['V']
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertFalse(causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs))

    def test_constructive_backdoor_criterion_should_not_hold_descendent_of_proper_causal_path(self):
        """ Test whether the constructive criterion holds when the adjustment set Z contains a descendent of a variable
        on a proper causal path between X and Y. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ['X1', 'X2'], ['Y'], ['D1']
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertFalse(causal_dag.constructive_backdoor_criterion(proper_backdoor_graph, xs, ys, zs))

    def test_is_min_adjustment_for_min_adjustment(self):
        """ Test whether is_min_adjustment can correctly test whether the minimum adjustment set is minimal."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ['X1', 'X2'], ['Y'], {'Z'}
        self.assertTrue(causal_dag.adjustment_set_is_minimal(xs, ys, zs))

    def test_is_min_adjustment_for_not_min_adjustment(self):
        """ Test whether is_min_adjustment can correctly test whether the minimum adjustment set is not minimal."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ['X1', 'X2'], ['Y'], {'Z', 'V'}
        self.assertFalse(causal_dag.adjustment_set_is_minimal(xs, ys, zs))

    def test_is_min_adjustment_for_invalid_adjustment(self):
        """ Test whether is min_adjustment can correctly identify that the minimum adjustment set is invalid."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ['X1', 'X2'], ['Y'], set()
        self.assertRaises(ValueError, causal_dag.adjustment_set_is_minimal, xs, ys, zs)

    def test_get_ancestor_graph_of_causal_dag(self):
        """ Test whether get_ancestor_graph converts a CausalDAG to the correct ancestor graph."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ['X1', 'X2'], ['Y']
        ancestor_graph = causal_dag.get_ancestor_graph(xs, ys)
        self.assertEqual(list(ancestor_graph.graph.nodes), ['X1', 'X2', 'D1', 'Y', 'Z'])
        self.assertEqual(list(ancestor_graph.graph.edges), [('X1', 'X2'), ('X2', 'D1'), ('D1', 'Y'), ('Z', 'X2'),
                                                            ('Z', 'Y')])

    def test_get_ancestor_graph_of_proper_backdoor_graph(self):
        """ Test whether get_ancestor_graph converts a CausalDAG to the correct ancestor graph."""
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ['X1', 'X2'], ['Y']
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        ancestor_graph = proper_backdoor_graph.get_ancestor_graph(xs, ys)
        self.assertEqual(list(ancestor_graph.graph.nodes), ['X1', 'X2', 'D1', 'Y', 'Z'])
        self.assertEqual(list(ancestor_graph.graph.edges), [('X1', 'X2'), ('D1', 'Y'), ('Z', 'X2'), ('Z', 'Y')])

    def test_enumerate_minimal_adjustment_sets(self):
        """ Test whether enumerate_minimal_adjustment_sets lists all possible minimum sized adjustment sets for a
        CausalDAG. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys = ['X1', 'X2'], ['Y']
        adjustment_sets = causal_dag.enumerate_minimal_adjustment_sets(xs, ys)
        print(list(adjustment_sets))

    def tearDown(self) -> None:
        os.remove(self.dag_dot_path)


class TestUndirectedGraphAlgorithms(unittest.TestCase):

    def setUp(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edges_from([('a', 2), ('a', 3), (2, 4), (3, 5), (3, 4), (4, 'b'), (5, 'b')])
        self.treatment_node = 'a'
        self.outcome_node = 'b'
        self.treatment_node_set = {'a'}
        self.outcome_node_set = set(nx.neighbors(self.graph, 'b'))
        self.outcome_node_set.add('b')

    def test_close_separator(self):
        result = close_separator(self.graph, self.treatment_node, self.outcome_node, self.treatment_node_set)
        self.assertEqual({2, 3}, result)

    def test_list_all_min_sep(self):
        min_separators = list(list_all_min_sep(self.graph, self.treatment_node, self.outcome_node,
                                               self.treatment_node_set, self.outcome_node_set))

        # Convert list of sets to set of frozen sets for comparison
        min_separators = set(frozenset(min_separator) for min_separator in min_separators)
        self.assertEqual({frozenset({2, 3}), frozenset({3, 4}), frozenset({4, 5})}, min_separators)


if __name__ == '__main__':
    unittest.main()
