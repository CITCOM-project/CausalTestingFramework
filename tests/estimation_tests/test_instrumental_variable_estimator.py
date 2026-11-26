"""
This module tests the instrumental variable estimator.
"""

import numpy as np
import pandas as pd
import pytest

from causal_testing.estimation.instrumental_variable_estimator import InstrumentalVariableEstimator


@pytest.fixture(name="df")
def _df():
    """
    Noiseless linear data where X = 2Z and Y = 2X.
    """
    z = np.linspace(0, 10)
    x = 2 * z
    y = 2 * x
    return pd.DataFrame({"Z": z, "X": x, "Y": y})


@pytest.fixture(name="iv_estimator")
def _iv_estimator():
    """
    Fresh estimator for each test, using Z as an instrument for the effect of X on Y.
    """
    return InstrumentalVariableEstimator(
        treatment_variable="X",
        outcome_variable="Y",
        instrument="Z",
    )


def test_estimate_coefficient(iv_estimator, df):
    """
    Test we get the correct coefficient (unit ATE).
    """
    effect_estimate = iv_estimator.estimate_coefficient(df)
    assert effect_estimate.effect_estimate[0] == 2
    assert effect_estimate.ci_low[0] == 2
    assert effect_estimate.ci_high[0] == 2


def test_to_dict(iv_estimator):
    """
    Test that the estimator is correctly serialised to a dictionary.
    """
    assert iv_estimator.to_dict() == {
        "name": "InstrumentalVariableEstimator",
        "treatment_variable": "X",
        "outcome_variable": "Y",
        "alpha": 0.05,
        "instrument": "Z",
        "bootstrap_size": 100,
    }
