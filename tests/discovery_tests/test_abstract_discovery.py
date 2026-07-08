"""
This module tests common causal discovery functionality provided within the abstract_discovery module.
"""

import unittest
import pandas as pd
from tempfile import TemporaryDirectory
import os

from causal_testing.discovery.abstract_discovery import TestResult, Discovery, simple_cycle, effect_direction
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output


class AbstractDiscovery(Discovery):
    """
    Minimal concrete implementation of abstract "Discovery" class.
    """

    def discover(self):
        """
        Minimal implementation of the abstract method.
        """
        return CausalDAG()


class TestAbstractHillClimber(unittest.TestCase):
    def setUp(self) -> None:
        base_test_case = BaseTestCase(Input("A", float), Output("B", float))
        self.estimator = LinearRegressionEstimator(
            df=pd.DataFrame({"A": [1, 2], "B": [4, 5]}),
            base_test_case=base_test_case,
            treatment_value=1,
            control_value=0,
            adjustment_set={},
        )

    def test_simple_cycle(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        self.assertEqual(simple_cycle(dag), [("A", "B"), ("B", "C"), ("C", "A")])

    def test_simple_cycle_no_cycles(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        self.assertEqual(simple_cycle(dag), [])

    def test_effect_direction_positive(self):
        ctr = CausalTestResult(
            estimator=self.estimator,
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(5.05)),
        )
        self.assertEqual(effect_direction(ctr), "positive")

    def test_effect_direction_negative(self):
        ctr = CausalTestResult(
            estimator=self.estimator,
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(-5.05)),
        )
        self.assertEqual(effect_direction(ctr), "negative")

    def test_effect_direction_none(self):
        ctr = CausalTestResult(
            estimator=self.estimator,
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0)),
        )
        self.assertEqual(effect_direction(ctr), None)

    def test_include_edge_wildcard(self):
        abstract_discovery = AbstractDiscovery(
            df=pd.DataFrame(columns=["x_1", "x_2", "x_3", "y_1", "y_2", "y_3", "z_1", "z_2"]),
            include_edges=[("x_.*", "y_1")],
        )
        self.assertEqual(abstract_discovery.include_edges, [(f"x_{n}", "y_1") for n in range(1, 4)])

    def test_include_edge_cycle(self):
        with self.assertRaises(ValueError):
            AbstractDiscovery(
                df=pd.DataFrame(columns=["x_1", "x_2", "x_3", "y_1", "y_2", "y_3", "z_1", "z_2"]),
                include_edges=[("x_1", "y_1"), ("y_1", "x_1")],
            )

    def test_exclude_edge_wildcard(self):
        abstract_discovery = AbstractDiscovery(
            df=pd.DataFrame(columns=["x_1", "x_2", "x_3", "y_1", "y_2", "y_3", "z_1", "z_2"]),
            exclude_edges=[("x_.*", "y_1")],
        )
        self.assertEqual(abstract_discovery.exclude_edges, [(f"x_{n}", "y_1") for n in range(1, 4)])

    def test_remove_cycles(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        dag.add_edge("C", "A", ignore_cycles=True)
        self.assertFalse(dag.is_acyclic(), "A -> B -> C -> A should form a cycle.")

        abstract_discovery = AbstractDiscovery(pd.DataFrame())
        abstract_discovery.remove_cycles(dag)
        self.assertTrue(dag.is_acyclic())

    def test_remove_cycles_respects_include_edges(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        dag.add_edge("C", "A", ignore_cycles=True)

        include_edges = {("A", "B"), ("B", "C")}
        abstract_discovery = AbstractDiscovery(pd.DataFrame(columns=dag.nodes), include_edges=include_edges)

        abstract_discovery.remove_cycles(dag)
        self.assertTrue(dag.is_acyclic())
        for edge in include_edges:
            self.assertTrue(edge in dag.edges, f"{edge} not in {dag.edges}")

    def test_remove_cycles_no_cycles_present(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B")])

        abstract_discovery = AbstractDiscovery(pd.DataFrame())
        abstract_discovery.remove_cycles(dag)
        self.assertEqual(len(dag.edges()), 1)

    def test_remove_cycles_multiple_cycles(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("C", "D"), ("B", "A"), ("D", "C")])

        abstract_discovery = AbstractDiscovery(pd.DataFrame())
        abstract_discovery.remove_cycles(dag)
        self.assertEqual(len(dag.edges()), 2)
        self.assertTrue(dag.has_edge("A", "B") or dag.has_edge("B", "A"))
        self.assertTrue(dag.has_edge("C", "D") or dag.has_edge("D", "C"))

    def test_write_dot(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("C", "D"), ("E", "F")])
        dag.test_results = pd.DataFrame(
            [  # Edges
                {"treatment": "A", "outcome": "B", "result": TestResult.PASS},
                {"treatment": "C", "outcome": "D", "result": TestResult.FAIL},
                {"treatment": "E", "outcome": "F", "result": TestResult.INESTIMABLE},
                # Independences
                {"treatment": "A", "outcome": "C", "result": TestResult.PASS},
                {"treatment": "A", "outcome": "D", "result": TestResult.FAIL},
                {"treatment": "A", "outcome": "E", "result": TestResult.INESTIMABLE},
            ]
        )
        abstract_discovery = AbstractDiscovery(pd.DataFrame())
        with TemporaryDirectory() as tmp:
            abstract_discovery.write_dot(dag, os.path.join(tmp, "dag.dot"))
            dag2 = CausalDAG(os.path.join(tmp, "dag.dot"))
            self.assertEqual(dag.nodes, dag2.nodes)

    def test_write_dot_invalid_edge_outcome(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("C", "D"), ("E", "F")])
        dag.test_results = pd.DataFrame(
            [  # Edges
                {"treatment": "A", "outcome": "B", "result": None},
            ]
        )
        abstract_discovery = AbstractDiscovery(pd.DataFrame())
        with self.assertRaises(ValueError):
            abstract_discovery.write_dot(dag, "dag.dot")

    def test_write_dot_invalid_independence_outcome(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("C", "D"), ("E", "F")])
        dag.test_results = pd.DataFrame(
            [  # Edges
                {"treatment": "A", "outcome": "C", "result": None},
            ]
        )
        abstract_discovery = AbstractDiscovery(pd.DataFrame())
        with self.assertRaises(ValueError):
            abstract_discovery.write_dot(dag, "dag.dot")

    def test_evaluate_tests(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

        dag = CausalDAG()
        dag.add_nodes_from(scarf_df.columns)
        dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])

        abstract_discovery = AbstractDiscovery(scarf_df)
        test_results = abstract_discovery.evaluate_tests(dag)
        expected_results = pd.DataFrame(
            [
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "large_gauge",
                    "effect": "positive",
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "large_gauge",
                    "outcome": "length_in",
                    "effect": "positive",
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "color",
                    "effect": None,
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "color",
                    "outcome": "length_in",
                    "effect": None,
                },
                {
                    "result": TestResult.FAIL,
                    "expected_effect": "SomeEffect",
                    "treatment": "length_in",
                    "outcome": "completed",
                    "effect": "negative",
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "large_gauge",
                    "outcome": "color",
                    "effect": None,
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "color",
                    "outcome": "large_gauge",
                    "effect": None,
                },
                {
                    "result": TestResult.FAIL,
                    "expected_effect": "SomeEffect",
                    "treatment": "large_gauge",
                    "outcome": "completed",
                    "effect": "positive",
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "color",
                    "outcome": "completed",
                    "effect": None,
                },
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "completed",
                    "outcome": "color",
                    "effect": None,
                },
            ]
        )
        pd.testing.assert_frame_equal(test_results, expected_results)
