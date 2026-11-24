"""
This module tests the ExperimentalEstimator.
"""

import pytest

from causal_testing.estimation.experimental_estimator import ExperimentalEstimator


class SystemUnderTest:
    """
    Basic example of a system under test.
    """

    def run(self, x):
        return x * 2


class ConcreteExperimentalEstimator(ExperimentalEstimator):
    """
    Concrete experimental estimator class which integrates with the system under test.
    """

    def run_system(self, configuration: dict):
        """
        Sets up the system under test, runs with the given configuration, and returns the result in the correct format.
        :param configuration: The configuration.
        :returns: Dictionary with the output.
        """
        sut = SystemUnderTest()
        return {"Y": sut.run(configuration["X"])}


@pytest.fixture(name="estimator")
def _estimator():
    """
    Fresh estimator for each test.
    """
    return ConcreteExperimentalEstimator(
        treatment_variable="X",
        outcome_variable="Y",
        treatment_value=2,
        control_value=1,
        adjustment_config={},
        alpha=0.05,
        repeats=200,
    )


def test_estimate_ate(estimator):
    effect_estimate = estimator.estimate_ate()
    assert effect_estimate.effect_estimate["X"] == 2
    assert effect_estimate.ci_low["X"] == 2
    assert effect_estimate.ci_high["X"] == 2


def test_estimate_risk_ratio(estimator):
    effect_estimate = estimator.estimate_risk_ratio()
    assert effect_estimate.effect_estimate["X"] == 2
    assert effect_estimate.ci_low["X"] == 2
    assert effect_estimate.ci_high["X"] == 2
