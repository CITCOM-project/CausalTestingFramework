import unittest
import os
from causal_testing.specification.causal_specification import CausalDAG


class TestCausalDAG(unittest.TestCase):

    def setUp(self) -> None:
        self.dag_dot_path = 'temp/dag.dot'
        dag_dot = """digraph G { A -> B; B -> C; D -> A; D -> C}"""
        f = open(self.dag_dot_path, 'w')
        f.write(dag_dot)
        f.close()

    def test_something(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        print(causal_dag)

    def tearDown(self) -> None:
        os.remove('temp/dag.dot')


if __name__ == '__main__':
    unittest.main()
