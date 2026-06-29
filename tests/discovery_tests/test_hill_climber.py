import unittest

from causal_testing.discovery.hill_climber import simple_cycle, remove_cycles
from causal_testing.specification.causal_dag import CausalDAG

class TestHillClimber(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple_cycle_normal(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        super(CausalDAG, dag).add_edge("C", "A")
        
        cycle = simple_cycle(dag)
        self.assertEqual(set(cycle), {("A", "B"), ("B", "C"), ("C", "A")})

    def test_simple_cycle_no_cycles(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        
        cycle = simple_cycle(dag)
        self.assertEqual(set(cycle), set())

    def test_remove_cycles_normal(self):
        dag = CausalDAG()
        dag.add_nodes_from(['A', 'B', 'C'])
        dag.add_edges_from([("A", "B"), ("B", "C")])
        super(CausalDAG, dag).add_edge("C", "A")

        remove_cycles(dag, included_edges=set())
        self.assertEqual(len(dag.edges()), 2)

    def test_remove_cycles_respects_included_edges(self):
        dag = CausalDAG()
        dag.add_nodes_from(['A', 'B', 'C'])
        dag.add_edges_from([("A", "B"), ("B", "C")])
        super(CausalDAG, dag).add_edge("C", "A")
        
        included_edges = {('A', 'B'), ('B', 'C')}        
        remove_cycles(dag, included_edges)
        self.assertFalse(dag.has_edge("C", "A"))

    def test_remove_cycles_no_cycles_present(self):
        dag = CausalDAG()
        dag.add_nodes_from(['A', 'B'])
        dag.add_edges_from([("A", "B")])
        
        remove_cycles(dag, included_edges=set())        
        self.assertEqual(len(dag.edges()), 1)

    def test_remove_cycles_multiple_cycles(self):
        dag = CausalDAG()
        dag.add_nodes_from(['A', 'B', 'C', 'D'])
        dag.add_edges_from([("A", "B"), ("C", "D")])
        super(CausalDAG, dag).add_edge("B", "A")
        super(CausalDAG, dag).add_edge("D", "C")
                
        remove_cycles(dag, included_edges=set())        
        self.assertEqual(len(dag.edges()), 2)
        self.assertTrue(dag.has_edge("A", "B") or dag.has_edge("B", "A"))
        self.assertTrue(dag.has_edge("C", "D") or dag.has_edge("D", "C"))

    def test_estimate_effect(self):
        # Test the estimate_effect function
        pass

    def test_evaluate_tests(self):
        # Test the evaluate_dag function
        pass

    def test_normalize_counts(self):
        # Test the normalize_counts function
        pass

    def test_evaluate_fitness(self):
        # Test the evaluate_fitness function
        pass

    def test_evolve_dag(self):
        # Test the evolve_dag function
        pass

    def tearDown(self):
        pass