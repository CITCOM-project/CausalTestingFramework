import unittest
import pandas as pd
from causal_testing.discovery.hill_climber_discovery import HillClimberDiscovery
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.discovery.abstract_discovery import TestResult, Discovery, simple_cycle, effect_direction


class TestHillClimber(unittest.TestCase):

    def test_sum_test_outcomes(self):
        test_results = pd.DataFrame(
            [
                {
                    "result": TestResult.PASS,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "large_gauge",
                    "effect": "positive",
                },
                {
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "large_gauge",
                    "outcome": "length_in",
                    "effect": None,
                },
                {
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "color",
                    "effect": None,
                },
                {
                    "result": TestResult.INESTIMABLE,
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
                    "result": TestResult.INESTIMABLE,
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
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "completed",
                    "outcome": "color",
                    "effect": None,
                },
            ]
        )
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
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "large_gauge",
                    "outcome": "length_in",
                    "effect": None,
                },
                {
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "length_in",
                    "outcome": "color",
                    "effect": None,
                },
                {
                    "result": TestResult.INESTIMABLE,
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
                    "result": TestResult.INESTIMABLE,
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
                    "result": TestResult.INESTIMABLE,
                    "expected_effect": "NoEffect",
                    "treatment": "completed",
                    "outcome": "color",
                    "effect": None,
                },
            ]
        )
