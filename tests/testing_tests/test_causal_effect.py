"""
Test the EffectEstimate class, the CausalEffect subclasses, and E-value estimation.
"""

import re

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


@pytest.mark.parametrize(
    ["causal_effect", "effect_measure", "estimate", "ci_low", "ci_high", "expected"],
    [
        pytest.param(Positive(), "ate", 5.05, 5, 6, True, id="positive_ate_pass"),
        pytest.param(Positive(), "risk_ratio", 5.05, 5, 6, True, id="positive_risk_ratio_pass"),
        pytest.param(Positive(), "ate", 0, -1, 1, False, id="positive_fail"),
        pytest.param(Positive(), "risk_ratio", 1, 0.8, 1.2, False, id="positive_risk_ratio_fail"),
        pytest.param(Negative(), "ate", -5.05, -6, -5, True, id="negative_ate_pass"),
        pytest.param(Negative(), "risk_ratio", 0.2, 0.1, 0.5, True, id="negative_risk_ratio_pass"),
        pytest.param(Negative(), "ate", 0, -1, 1, False, id="negative_fail"),
        pytest.param(Negative(), "risk_ratio", 1, 0.8, 1.2, False, id="negative_risk_ratio_fail"),
    ],
)
def test_effect_direction(
    causal_effect, effect_measure, estimate, ci_low, ci_high, expected
):  # pylint: disable=too-many-positional-arguments
    """Test that Positive and Negative pass only when the CI lies entirely on the correct side of the null value."""
    effect_estimate = EffectEstimate(
        effect_measure=effect_measure,
        effect_estimate=pd.Series(estimate),
        ci_low=pd.Series(ci_low),
        ci_high=pd.Series(ci_high),
    )
    assert causal_effect.apply(effect_estimate) == expected


def test_exact_value_pass():
    """Test that ExactValue passes when the estimate is within atol of the value."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series(5.05))
    assert ExactValue(value=5, atol=0.1).apply(effect_estimate)


def test_exact_value_to_dict():
    """Test that ExactValue converts to a dict."""
    assert ExactValue(value=5.01, ci_low=5.0, ci_high=5.08).to_dict(), {
        "name": "ExactValue",
        "effect_type": "direct",
        "value": 5.01,
        "ci_high": 5.08,
        "ci_low": 5.0,
    }


def test_exact_value_categorical_pass():
    """Test that ExactValue passes for categorical estimates within atol of the values."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series({"color[T.red]": 5.05, "color[T.blue]": 4.03})
    )
    assert ExactValue(value=pd.Series({"color[T.red]": 5, "color[T.blue]": 4}), atol=0.1).apply(effect_estimate)


def test_exact_value_pass_ci():
    """Test that ExactValue passes when the estimate has a CI but ExactValue does not."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4), ci_high=pd.Series(6)
    )
    assert ExactValue(value=5, atol=0.1).apply(effect_estimate)


def test_exact_value_ci_pass_ci():
    """Test that ExactValue with a CI passes when the estimate and its CI are within bounds."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.1), ci_high=pd.Series(5.9)
    )
    assert ExactValue(value=5, atol=0.05, ci_low=4, ci_high=6).apply(effect_estimate)


def test_exact_value_ci_fail_ci():
    """Test that ExactValue with a CI fails when the estimate is outside atol."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.1), ci_high=pd.Series(5.9)
    )
    assert not ExactValue(value=5, atol=0.04, ci_low=4, ci_high=6).apply(effect_estimate)


def test_exact_value_fail():
    """Test that ExactValue fails when the estimate is outside atol."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series(0))
    assert not ExactValue(value=5, atol=0.1).apply(effect_estimate)


@pytest.mark.parametrize(
    ["kwargs", "message"],
    [
        pytest.param(
            {"value": 5, "atol": -0.1},
            "Tolerance must be an absolute (positive) value.",
            id="invalid_atol",
        ),
        pytest.param(
            {"value": 5, "ci_low": -0.1},
            "If specifying confidence intervals, must specify `ci_low` and `ci_high` parameters.",
            id="unspecified_ci_high",
        ),
        pytest.param(
            {"value": 5, "ci_high": -0.1},
            "If specifying confidence intervals, must specify `ci_low` and `ci_high` parameters.",
            id="unspecified_ci_low",
        ),
        pytest.param(
            {"value": 5, "ci_low": 6, "ci_high": 7, "atol": 0.05},
            "Specified value falls outside the specified confidence intervals.",
            id="invalid_ci_range",
        ),
        pytest.param(
            {"value": 1000, "ci_low": 999, "ci_high": 1001, "atol": 50},
            "Arithmetic tolerance falls outside the confidence intervals. Try specifying wider intervals or a value of "
            "atol smaller than the current value 50.",
            id="invalid_ci_atol",
        ),
    ],
)
def test_exact_value_invalid_construction(kwargs, message):
    """Test that constructing an ExactValue with invalid parameters raises an error."""
    with pytest.raises(ValueError, match=re.escape(message)):
        ExactValue(**kwargs)


@pytest.mark.parametrize(
    "causal_effect",
    [
        pytest.param(SomeEffect(), id="SomeEffect"),
        pytest.param(NoEffect(), id="NoEffect"),
        pytest.param(Positive(), id="Positive"),
        pytest.param(Negative(), id="Negative"),
    ],
)
def test_invalid(causal_effect):
    """Test that applying an effect to an estimate with an invalid effect measure raises an error."""
    effect_estimate = EffectEstimate(
        effect_measure="invalid", effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
    )
    with pytest.raises(ValueError, match=re.escape("Test Value type invalid is not valid for this CausalEffect")):
        causal_effect.apply(effect_estimate)


@pytest.mark.parametrize("effect_measure", ["coefficient", "ate", "risk_ratio"])
def test_some_effect_pass(effect_measure):
    """Test that SomeEffect passes and NoEffect fails for an estimate whose CI excludes the null value."""
    effect_estimate = EffectEstimate(
        effect_measure=effect_measure, effect_estimate=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
    )
    assert SomeEffect().apply(effect_estimate)
    assert not NoEffect().apply(effect_estimate)


def test_some_effect_fail():
    """Test that SomeEffect fails and NoEffect passes when the CI spans zero."""
    effect_estimate = EffectEstimate(
        effect_measure="ate", effect_estimate=pd.Series(0), ci_low=pd.Series(-0.1), ci_high=pd.Series(0.2)
    )
    assert not SomeEffect().apply(effect_estimate)
    assert NoEffect().apply(effect_estimate)


@pytest.mark.parametrize(
    ["risk_ratio", "expected"],
    [
        pytest.param(1.5, 2.366, id="positive_risk_ratio"),
        pytest.param(0.8, 1.809, id="negative_risk_ratio"),
    ],
)
def test_e_value(risk_ratio, expected):
    """Test the E-value for risk ratios either side of 1."""
    e_value = CausalValidator().estimate_e_value(risk_ratio)
    assert round(e_value, 4) == expected


@pytest.mark.parametrize(
    ["risk_ratio", "ci", "expected"],
    [
        pytest.param(1.5, [1.2, 1.8], 1.6899, id="positive_risk_ratio"),
        pytest.param(0.8, [0.2, 0.9], 1.4625, id="negative_risk_ratio"),
    ],
)
def test_e_value_using_ci(risk_ratio, ci, expected):
    """Test the CI-based E-value for risk ratios either side of 1."""
    e_value = CausalValidator().estimate_e_value_using_ci(risk_ratio, ci)
    assert round(e_value, 4) == expected


@pytest.mark.parametrize(
    "causal_effect",
    [
        pytest.param(Positive(), id="Positive"),
        pytest.param(Negative(), id="Negative"),
    ],
)
def test_multiple_value_exception_caught(causal_effect):
    """Test that Positive and Negative raise an error for a multi-valued effect estimate."""
    effect_estimate = EffectEstimate(effect_measure="ate", effect_estimate=pd.Series([0, 1]))
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{type(causal_effect).__name__} Effects are currently only supported on single float datatypes"
        ),
    ):
        causal_effect.apply(effect_estimate)
