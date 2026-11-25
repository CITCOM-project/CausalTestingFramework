"""
Test the CausalTestResult class.
"""

import pandas as pd

from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.testing.causal_test_result import CausalTestResult


def test_effect_direction_positive():
    """Test that a strictly positive effect estimate and CI gives a positive effect direction."""
    result = CausalTestResult(
        outcome=None,
        effect_estimate=EffectEstimate(
            effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(5), ci_high=pd.Series(6)
        ),
    )
    assert result.effect_direction() == "Positive"


def test_effect_direction_negative():
    """Test that a strictly negative effect estimate and CI gives a negative effect direction."""
    result = CausalTestResult(
        outcome=None,
        effect_estimate=EffectEstimate(
            effect_measure="ate", effect_estimate=pd.Series(-5.05), ci_low=pd.Series(-6), ci_high=pd.Series(-5)
        ),
    )
    assert result.effect_direction() == "Negative"


def test_effect_direction_none():
    """Test that a CI spanning zero gives no effect direction."""
    result = CausalTestResult(
        outcome=None,
        effect_estimate=EffectEstimate(
            effect_measure="ate", effect_estimate=pd.Series(0), ci_low=pd.Series(-1), ci_high=pd.Series(1)
        ),
    )
    assert result.effect_direction() == ""


def test_effect_direction_categorical():
    """Test that categorical CIs spanning zero give no effect direction."""
    result = CausalTestResult(
        outcome=None,
        effect_estimate=EffectEstimate(
            effect_measure="ate",
            effect_estimate=pd.Series({"color[T.RED]": -5, "color[T.BLUE]": -4}),
            ci_low=pd.Series({"color[T.RED]": -4, "color[T.BLUE]": -1}),
            ci_high=pd.Series({"color[T.RED]": 5, "color[T.BLUE]": 4}),
        ),
    )
    assert result.effect_direction() == ""
