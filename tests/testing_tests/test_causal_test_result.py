"""
Test the CausalTestResult class.
"""

import pandas as pd
import pytest

from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.testing.causal_test_result import CausalTestResult


@pytest.mark.parametrize(
    ["estimate", "ci_low", "ci_high", "expected_direction"],
    [
        pytest.param(pd.Series(5.05), pd.Series(5), pd.Series(6), "Positive", id="positive"),
        pytest.param(pd.Series(-5.05), pd.Series(-6), pd.Series(-5), "Negative", id="negative"),
        pytest.param(pd.Series(0), pd.Series(-1), pd.Series(1), "", id="none"),
        pytest.param(
            pd.Series({"color[T.RED]": -5, "color[T.BLUE]": -4}),
            pd.Series({"color[T.RED]": -4, "color[T.BLUE]": -1}),
            pd.Series({"color[T.RED]": 5, "color[T.BLUE]": 4}),
            "",
            id="categorical",
        ),
    ],
)
def test_effect_direction(estimate, ci_low, ci_high, expected_direction):
    """
    Test that the effect direction is positive or negative when the CI lies entirely above or below zero, and empty
    when it spans zero.
    """
    result = CausalTestResult(
        outcome=None,
        effect_estimate=EffectEstimate(effect_measure="ate", effect_estimate=estimate, ci_low=ci_low, ci_high=ci_high),
    )
    assert result.effect_direction() == expected_direction
