"""
Test the EffectEstimate class, the CausalEffect subclasses, and E-value estimation.
"""

import pandas as pd
import pytest

from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.testing.causal_effect import ExactValue, Negative, NoEffect, Positive, SomeEffect
from causal_testing.utils.validation import CausalValidator


def test_effect_estimate_to_dict():
    """Test that an EffectEstimate with a CI converts to a dict."""
    effect_estimate = EffectEstimate(
        effect_measure="ate",
        effect_estimate=pd.Series({"A": 1}),
        ci_low=pd.Series({"A": 0.1}),
        ci_high=pd.Series({"A": 1.2}),
    )
    assert effect_estimate.to_dict() == {
        "effect_measure": "ate",
        "effect_estimate": {"A": 1},
        "ci_low": {"A": 0.1},
        "ci_high": {"A": 1.2},
    }


def test_effect_estimate_to_dict_no_ci():
    """Test that an EffectEstimate without a CI converts to a dict with no CI keys."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series({"A": 1}))
    assert effect_estimate.to_dict() == {"effect_measure": "ate", "effect_estimate": {"A": 1}}


def test_Positive_ate_pass():  # pylint: disable=invalid-name
    """Test that Positive passes for a positive ATE."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(5), ci_high=pd.Series(6)
    )
    assert Positive().apply(effect_estimate)


def test_Positive_risk_ratio_pass():  # pylint: disable=invalid-name
    """Test that Positive passes for a risk ratio greater than 1."""
    effect_estimate = EffectEstimate(
        effect_measure="risk_ratio", effect_estimate=pd.Series(5.05), ci_low=pd.Series(5), ci_high=pd.Series(6)
    )
    assert Positive().apply(effect_estimate)


def test_Positive_fail():  # pylint: disable=invalid-name
    """Test that Positive fails when the CI spans zero."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(0), ci_low=pd.Series(-1), ci_high=pd.Series(1)
    )
    assert not Positive().apply(effect_estimate)


def test_Negative_ate_pass():  # pylint: disable=invalid-name
    """Test that Negative passes for a negative ATE."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(-5.05), ci_low=pd.Series(-6), ci_high=pd.Series(-5)
    )
    assert Negative().apply(effect_estimate)


def test_Negative_risk_ratio_pass():  # pylint: disable=invalid-name
    """Test that Negative passes for a risk ratio less than 1."""
    effect_estimate = EffectEstimate(
        effect_measure="risk_ratio", effect_estimate=pd.Series(0.2), ci_low=pd.Series(0.1), ci_high=pd.Series(0.5)
    )
    assert Negative().apply(effect_estimate)


def test_Negative_fail():  # pylint: disable=invalid-name
    """Test that Negative fails when the CI spans zero."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(0), ci_low=pd.Series(-1), ci_high=pd.Series(1)
    )
    assert not Negative().apply(effect_estimate)


def test_exactValue_pass():  # pylint: disable=invalid-name
    """Test that ExactValue passes when the estimate is within atol of the value."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series(5.05))
    assert ExactValue(value=5, atol=0.1).apply(effect_estimate)


def test_exactValue_to_dict():  # pylint: disable=invalid-name
    """Test that ExactValue converts to a dict."""
    assert ExactValue(value=5.01, ci_low=5.0, ci_high=5.08).to_dict(), {
        "name": "ExactValue",
        "effect_type": "direct",
        "value": 5.01,
        "ci_high": 5.08,
        "ci_low": 5.0,
    }


def test_exactValue_categorical_pass():  # pylint: disable=invalid-name
    """Test that ExactValue passes for categorical estimates within atol of the values."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series({"color[T.red]": 5.05, "color[T.blue]": 4.03})
    )
    assert ExactValue(value=pd.Series({"color[T.red]": 5, "color[T.blue]": 4}), atol=0.1).apply(effect_estimate)


def test_exactValue_pass_ci():  # pylint: disable=invalid-name
    """Test that ExactValue passes when the estimate has a CI but ExactValue does not."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4), ci_high=pd.Series(6)
    )
    assert ExactValue(value=5, atol=0.1).apply(effect_estimate)


def test_exactValue_ci_pass_ci():  # pylint: disable=invalid-name
    """Test that ExactValue with a CI passes when the estimate and its CI are within bounds."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.1), ci_high=pd.Series(5.9)
    )
    assert ExactValue(value=5, atol=0.05, ci_low=4, ci_high=6).apply(effect_estimate)


def test_exactValue_ci_fail_ci():  # pylint: disable=invalid-name
    """Test that ExactValue with a CI fails when the estimate is outside atol."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.1), ci_high=pd.Series(5.9)
    )
    assert not ExactValue(value=5, atol=0.04, ci_low=4, ci_high=6).apply(effect_estimate)


def test_exactValue_fail():  # pylint: disable=invalid-name
    """Test that ExactValue fails when the estimate is outside atol."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series(0))
    assert not ExactValue(value=5, atol=0.1).apply(effect_estimate)


def test_invalid_atol():
    """Test that a negative atol raises an error."""
    with pytest.raises(ValueError):
        ExactValue(value=5, atol=-0.1)


def test_unspecified_ci_high():
    """Test that specifying ci_low without ci_high raises an error."""
    with pytest.raises(ValueError):
        ExactValue(value=5, ci_low=-0.1)


def test_unspecified_ci_low():
    """Test that specifying ci_high without ci_low raises an error."""
    with pytest.raises(ValueError):
        ExactValue(value=5, ci_high=-0.1)


def test_invalid_ci_range():
    """Test that a CI which does not contain the value raises an error."""
    with pytest.raises(ValueError):
        ExactValue(value=5, ci_low=6, ci_high=7, atol=0.05)


def test_invalid_ci_atol():
    """Test that an atol range which is not contained within the CI raises an error."""
    with pytest.raises(ValueError):
        ExactValue(value=1000, ci_low=999, ci_high=1001, atol=50)


def test_invalid():
    """Test that applying an effect to an estimate with an invalid effect measure raises an error."""
    effect_estimate = EffectEstimate(
        effect_measure="invalid", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
    )
    with pytest.raises(ValueError):
        SomeEffect().apply(effect_estimate)
    with pytest.raises(ValueError):
        NoEffect().apply(effect_estimate)
    with pytest.raises(ValueError):
        Positive().apply(effect_estimate)
    with pytest.raises(ValueError):
        Negative().apply(effect_estimate)


def test_someEffect_pass_coefficient():  # pylint: disable=invalid-name
    """Test that SomeEffect passes and NoEffect fails for a nonzero coefficient."""
    effect_estimate = EffectEstimate(
        effect_measure="coefficient", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
    )
    assert SomeEffect().apply(effect_estimate)
    assert not NoEffect().apply(effect_estimate)


def test_someEffect_pass_ate():  # pylint: disable=invalid-name
    """Test that SomeEffect passes and NoEffect fails for a nonzero ATE."""
    effect_estimate = EffectEstimate(
        effect_measure="coefficient", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
    )
    assert SomeEffect().apply(effect_estimate)
    assert not NoEffect().apply(effect_estimate)


def test_someEffect_pass_rr():  # pylint: disable=invalid-name
    """Test that SomeEffect passes and NoEffect fails for a risk ratio other than 1."""
    effect_estimate = EffectEstimate(
        effect_measure="coefficient", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
    )
    assert SomeEffect().apply(effect_estimate)
    assert not NoEffect().apply(effect_estimate)


def test_someEffect_fail():  # pylint: disable=invalid-name
    """Test that SomeEffect fails and NoEffect passes when the CI spans zero."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(0), ci_low=pd.Series(-0.1), ci_high=pd.Series(0.2)
    )
    assert not SomeEffect().apply(effect_estimate)
    assert NoEffect().apply(effect_estimate)


def test_positive_risk_ratio_e_value():
    """Test the E-value for a risk ratio greater than 1."""
    cv = CausalValidator()
    e_value = cv.estimate_e_value(1.5)
    assert round(e_value, 4) == 2.366


def test_positive_risk_ratio_e_value_using_ci():
    """Test the CI-based E-value for a risk ratio greater than 1."""
    cv = CausalValidator()
    e_value = cv.estimate_e_value_using_ci(1.5, [1.2, 1.8])
    assert round(e_value, 4) == 1.6899


def test_negative_risk_ratio_e_value():
    """Test the E-value for a risk ratio less than 1."""
    cv = CausalValidator()
    e_value = cv.estimate_e_value(0.8)
    assert round(e_value, 4) == 1.809


def test_negative_risk_ratio_e_value_using_ci():
    """Test the CI-based E-value for a risk ratio less than 1."""
    cv = CausalValidator()
    e_value = cv.estimate_e_value_using_ci(0.8, [0.2, 0.9])
    assert round(e_value, 4) == 1.4625


def test_multiple_value_exception_caught():
    """Test that Positive and Negative raise an error for a multi-valued effect estimate."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series([0, 1]))
    with pytest.raises(ValueError):
        Positive().apply(effect_estimate)
    with pytest.raises(ValueError):
        Negative().apply(effect_estimate)
