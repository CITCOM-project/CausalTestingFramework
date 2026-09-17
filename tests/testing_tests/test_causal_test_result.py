"""
Test the CausalTestResult class.
"""

import unittest

import pandas as pd

from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.testing.causal_test_result import CausalTestResult


class TestCausalTestCase(unittest.TestCase):

    def test_effect_direction_positive(self):
        result = CausalTestResult(
            outcome=None,
            effect_estimate=EffectEstimate(
                effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(5), ci_high=pd.Series(6)
            ),
        )
        self.assertEqual(result.effect_direction(), "positive")

    def test_effect_direction_negative(self):
        result = CausalTestResult(
            outcome=None,
            effect_estimate=EffectEstimate(
                effect_measure="ate", effect_estimate=pd.Series(-5.05), ci_low=pd.Series(-6), ci_high=pd.Series(-5)
            ),
        )
        self.assertEqual(result.effect_direction(), "negative")

    def test_effect_direction_none(self):
        result = CausalTestResult(
            outcome=None,
            effect_estimate=EffectEstimate(
                effect_measure="ate", effect_estimate=pd.Series(0), ci_low=pd.Series(-1), ci_high=pd.Series(1)
            ),
        )
        self.assertEqual(result.effect_direction(), "no effect")

    def test_effect_direction_categorical(self):
        result = CausalTestResult(
            outcome=None,
            effect_estimate=EffectEstimate(
                effect_measure="ate",
                effect_estimate=pd.Series({"color[T.RED]": -5, "color[T.BLUE]": -4}),
                ci_low=pd.Series({"color[T.RED]": -4, "color[T.BLUE]": -1}),
                ci_high=pd.Series({"color[T.RED]": 5, "color[T.BLUE]": 4}),
            ),
        )
        self.assertEqual(result.effect_direction(), "categorical")
