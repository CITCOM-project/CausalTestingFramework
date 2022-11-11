import unittest
import os

from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.metamorphic_relation import ShouldCause

class TestMetamorphicRelation(unittest.TestCase):

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        self.dag_dot_path = os.path.join(temp_dir_path, "dag.dot")
        dag_dot = """digraph DAG { rankdir=LR; Z -> X; X -> M; M -> Y; Z -> M; }"""
        with open(self.dag_dot_path, "w") as f:
            f.write(dag_dot)

    def test_metamorphic_relation(self):
        causal_dag = CausalDAG(self.dag_dot_path)
        for edge in causal_dag.graph.edges:
            (u, v) = edge
            should_cause_MR = ShouldCause(u, v, None, causal_dag)
            should_cause_MR.generate_follow_up(1, -10.0, 10.0, 1)
            print(should_cause_MR.tests)
