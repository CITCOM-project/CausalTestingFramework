import unittest
import os
import networkx as nx
from causal_testing.specification.causal_specification import CausalDAG


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
        os.remove('temp/dag.dot')


if __name__ == '__main__':
    unittest.main()
