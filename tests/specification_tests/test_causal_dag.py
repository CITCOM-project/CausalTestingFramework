import unittest
import os
import networkx as nx
from causal_testing.specification.causal_specification import CausalDAG


class TestCausalDAG(unittest.TestCase):
    def setUp(self) -> None:
        self.dag_dir = "temp"
        if not os.path.exists(self.dag_dir):
            os.makedirs(self.dag_dir)
        self.dag_dot_path = f"{self.dag_dir}/dag.dot"
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C}"""
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_valid_causal_dag(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        assert list(causal_dag.graph.nodes) == ["A", "B", "C", "D"] and list(
            causal_dag.graph.edges
        ) == [("A", "B"), ("B", "C"), ("D", "A"), ("D", "C")]

    def test_invalid_causal_dag(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        self.assertRaises(nx.HasACycle, causal_dag.add_edge, "C", "A")

    def test_empty_casual_dag(self):
        causal_dag = CausalDAG()
        assert list(causal_dag.graph.nodes) == [] and list(causal_dag.graph.edges) == []

    def tearDown(self) -> None:
        os.remove(self.dag_dot_path)


class TestGraphTransformations(unittest.TestCase):
    def setUp(self) -> None:
        self.dag_dir = "temp"
        if not os.path.exists(self.dag_dir):
            os.makedirs(self.dag_dir)
        self.dag_dot_path = f"{self.dag_dir}/dag.dot"
        dag_dot = (
            """digraph G { X1->X2;X2->V;X2->D1;X2->D2;D1->Y;D1->D2;Y->D3;Z->X2;Z->Y;}"""
        )
        f = open(self.dag_dot_path, "w")
        f.write(dag_dot)
        f.close()

    def test_proper_backdoor_graph(self):
        """
        Test whether converting a Causal DAG to a proper back-door graph works correctly.
        A proper back-door graph should remove the first edge from all proper causal paths from X to Y, where
        X is the set of treatments and Y is the set of outcomes.
        """
        causal_dag = CausalDAG(self.dag_dot_path)
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(
            ["X1", "X2"], ["Y"]
        )
        self.assertEqual(
            list(proper_backdoor_graph.graph.edges),
            [
                ("X1", "X2"),
                ("X2", "V"),
                ("X2", "D2"),
                ("D1", "D2"),
                ("D1", "Y"),
                ("Y", "D3"),
                ("Z", "X2"),
                ("Z", "Y"),
            ],
        )

    def test_constructive_backdoor_criterion_should_hold(self):
        """ Test whether the constructive criterion holds when it should. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["Z"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertTrue(
            causal_dag.constructive_backdoor_criterion(
                proper_backdoor_graph, xs, ys, zs
            )
        )

    def test_constructive_backdoor_criterion_should_not_hold_not_d_separator_in_proper_backdoor_graph(
        self,
    ):
        """ Test whether the constructive criterion holds when the adjustment set Z is not a d-separator in the proper
        back-door graph. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["V"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertFalse(
            causal_dag.constructive_backdoor_criterion(
                proper_backdoor_graph, xs, ys, zs
            )
        )

    def test_constructive_backdoor_criterion_should_not_hold_descendent_of_proper_causal_path(
        self,
    ):
        """ Test whether the constructive criterion holds when the adjustment set Z contains a descendent of a variable
        on a proper causal path between X and Y. """
        causal_dag = CausalDAG(self.dag_dot_path)
        xs, ys, zs = ["X1", "X2"], ["Y"], ["D1"]
        proper_backdoor_graph = causal_dag.get_proper_backdoor_graph(xs, ys)
        self.assertFalse(
            causal_dag.constructive_backdoor_criterion(
                proper_backdoor_graph, xs, ys, zs
            )
        )

    def tearDown(self) -> None:
        os.remove(self.dag_dot_path)


if __name__ == "__main__":
    unittest.main()
