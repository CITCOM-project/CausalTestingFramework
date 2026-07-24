import unittest

import pandas as pd

from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator


class TestLogisticRegressionEstimator(unittest.TestCase):
    """Test the logistic regression estimator against the scarf example from
    https://investigate.ai/regression/logistic-regression/.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_odds_ratio(self):
        logistic_regression_estimator = LogisticRegressionEstimator(
            treatment_variable="length_in",
            outcome_variable="completed",
            control_value=65,
            treatment_value=55,
            adjustment_set=set(),
        )
        effect_estimate = logistic_regression_estimator.estimate_unit_odds_ratio(self.scarf_df)
        self.assertEqual(round(effect_estimate.value.iloc[0], 4), 0.8948)
