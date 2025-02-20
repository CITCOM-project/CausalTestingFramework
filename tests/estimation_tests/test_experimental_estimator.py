import unittest
from causal_testing.estimation.experimental_estimator import ExperimentalEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output


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


class TestExperimentalEstimator(unittest.TestCase):
    """
    Test the experimental estimator.
    """

    def test_estimate_ate(self):
        estimator = ConcreteExperimentalEstimator(
            base_test_case=BaseTestCase(Input("X", float), Output("Y", float)),
            treatment_value=2,
            control_value=1,
            adjustment_set={},
            alpha=0.05,
            repeats=200,
        )
        ate, [ci_low, ci_high] = estimator.estimate_ate()
        self.assertEqual(ate["X"], 2)
        self.assertEqual(ci_low["X"], 2)
        self.assertEqual(ci_high["X"], 2)

    def test_estimate_risk_ratio(self):
        estimator = ConcreteExperimentalEstimator(
            base_test_case=BaseTestCase(Input("X", float), Output("Y", float)),
            treatment_value=2,
            control_value=1,
            adjustment_set={},
            effect_modifiers={},
            alpha=0.05,
            repeats=200,
        )
        rr, [ci_low, ci_high] = estimator.estimate_risk_ratio()
        self.assertEqual(rr["X"], 2)
        self.assertEqual(ci_low["X"], 2)
        self.assertEqual(ci_high["X"], 2)
