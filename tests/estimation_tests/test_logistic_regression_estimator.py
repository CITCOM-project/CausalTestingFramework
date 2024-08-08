import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.variable import Input
from causal_testing.utils.validation import CausalValidator
from causal_testing.specification.capabilities import TreatmentSequence
from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator


class TestLogisticRegressionEstimator(unittest.TestCase):
    """Test the logistic regression estimator against the scarf example from
    https://investigate.ai/regression/logistic-regression/.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_odds_ratio(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        odds = logistic_regression_estimator.estimate_unit_odds_ratio()
        self.assertEqual(round(odds, 4), 0.8948)
