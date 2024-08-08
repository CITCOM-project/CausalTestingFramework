import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.variable import Input
from causal_testing.utils.validation import CausalValidator
from causal_testing.specification.capabilities import TreatmentSequence

from causal_testing.estimation.cubic_spline_estimator import CubicSplineRegressionEstimator
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator

from tests.estimation_tests.test_linear_regression_estimator import TestLinearRegressionEstimator


class TestCubicSplineRegressionEstimator(TestLinearRegressionEstimator):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_program_11_3_cublic_spline(self):
        """Test whether the cublic_spline regression implementation produces the same results as program 11.3 (p. 162).
        https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2023/10/hernanrobins_WhatIf_30sep23.pdf
        Slightly modified as Hernan et al. use linear regression for this example.
        """

        df = self.chapter_11_df.copy()

        cublic_spline_estimator = CubicSplineRegressionEstimator("treatments", 1, 0, set(), "outcomes", 3, df)

        ate_1 = cublic_spline_estimator.estimate_ate_calculated()

        self.assertEqual(
            round(
                cublic_spline_estimator.model.predict({"Intercept": 1, "treatments": 90}).iloc[0],
                1,
            ),
            195.6,
        )

        cublic_spline_estimator.treatment_value = 2
        ate_2 = cublic_spline_estimator.estimate_ate_calculated()

        # Doubling the treatemebnt value should roughly but not exactly double the ATE
        self.assertNotEqual(ate_1[0] * 2, ate_2[0])
        self.assertAlmostEqual(ate_1[0] * 2, ate_2[0])
