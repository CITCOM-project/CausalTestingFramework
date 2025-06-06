import unittest

from causal_testing.estimation.cubic_spline_estimator import CubicSplineRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output

from tests.estimation_tests.test_linear_regression_estimator import load_chapter_11_df


class TestCubicSplineRegressionEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_program_11_3_cublic_spline(self):
        """Test whether the cublic_spline regression implementation produces the same results as program 11.3 (p. 162).
        https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2023/10/hernanrobins_WhatIf_30sep23.pdf
        Slightly modified as Hernan et al. use linear regression for this example.
        """

        df = load_chapter_11_df()

        base_test_case = BaseTestCase(Input("treatments", float), Output("outcomes", float))

        cublic_spline_estimator = CubicSplineRegressionEstimator(base_test_case, 1, 0, set(), 3, df)

        ate_1 = cublic_spline_estimator.estimate_ate_calculated()

        cublic_spline_estimator.treatment_value = 2
        ate_2 = cublic_spline_estimator.estimate_ate_calculated()

        # Doubling the treatemebnt value should roughly but not exactly double the ATE
        self.assertNotEqual(ate_1[0] * 2, ate_2[0])
        self.assertAlmostEqual(ate_1[0] * 2, ate_2[0])
