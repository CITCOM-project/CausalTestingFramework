"""
This module tests the Hill Climber Discovery algorithm.
"""

import unittest
import pandas as pd
from causal_testing.discovery.hill_climber_discovery import HillClimberDiscovery
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.discovery.abstract_discovery import simple_cycle
from causal_testing.testing.causal_test_result import TestOutcome


class TestHillClimber(unittest.TestCase):

    def test_sum_test_outcomes(self):
        test_results = pd.DataFrame(
            [
                {
                    "result": TestOutcome.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "large_gauge",
                    "effect": "positive",
                },
                {
                    "result": TestOutcome.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "large_gauge",
                    "outcome": "length_in",
                    "effect": None,
                },
                {
                    "result": TestOutcome.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "color",
                    "effect": None,
                },
                {
                    "result": TestOutcome.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "color",
                    "outcome": "length_in",
                    "effect": None,
                },
                {
                    "result": TestOutcome.FAIL,
                    "expected_effect": "SomeEffect",
                    "treatment": "length_in",
                    "outcome": "completed",
                    "effect": "negative",
                },
                {
                    "result": TestOutcome.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "large_gauge",
                    "outcome": "color",
                    "effect": None,
                },
                {
                    "result": TestOutcome.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "color",
                    "outcome": "large_gauge",
                    "effect": None,
                },
                {
                    "result": TestOutcome.FAIL,
                    "expected_effect": "SomeEffect",
                    "treatment": "large_gauge",
                    "outcome": "completed",
                    "effect": "positive",
                },
                {
                    "result": TestOutcome.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "color",
                    "outcome": "completed",
                    "effect": None,
                },
                {
                    "result": TestOutcome.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "completed",
                    "outcome": "color",
                    "effect": None,
                },
            ]
        )
        expected_results = {TestOutcome.PASS: 1.5, TestOutcome.FAIL: 2, TestOutcome.INESTIMABLE: 2.5}
        hill_climber = HillClimberDiscovery(pd.DataFrame())
        self.assertEqual(expected_results, hill_climber.sum_test_outcomes(test_results))

    def test_sum_test_outcomes_uninitialised(self):
        hill_climber = HillClimberDiscovery(pd.DataFrame())
        expected_results = {TestOutcome.PASS: 0, TestOutcome.FAIL: 0, TestOutcome.INESTIMABLE: 0}

        self.assertEqual(
            expected_results, hill_climber.sum_test_outcomes(pd.DataFrame(columns=["treatment", "outcome", "result"]))
        )

    def test_evaluate_fitness(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        dag = CausalDAG()
        dag.add_nodes_from(scarf_df.columns)
        dag.add_edges_from([("length_in", "completed"), ("large_gauge", "completed")])

        hill_climber = HillClimberDiscovery(scarf_df)
        fitness_values, problem_edges = hill_climber.evaluate_fitness(dag)
        expected_fitness_values = (4, -2, 0)
        expected_problem_edges = [
            ("length_in", "completed"),
            ("large_gauge", "completed"),
        ]

        self.assertEqual(fitness_values, expected_fitness_values)
        self.assertEqual(problem_edges, expected_problem_edges)

    def test_discovery_edges(self):
        scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")
        hill_climber = HillClimberDiscovery(
            scarf_df,
            include_edges=[("length_in", "completed")],
            exclude_edges=[("color", "length_in")],
            max_iterations=10,
        )
        dag = hill_climber.discover()
        self.assertTrue(
            ("length_in", "completed") in dag.edges, f"Expected ('length_in', 'completed') to be in {dag.edges}"
        )
        self.assertFalse(
            ("color", "completed") in dag.edges, f"Expected ('color', 'completed') NOT to be in {dag.edges}"
        )
