import unittest
import pandas as pd
import numpy as np
from causal_testing.testing.estimators import LinearRegressionEstimator


class TestLinearRegressionEstimator(unittest.TestCase):

    """
    Test the linear regression estimator against the programming exercises in Section 2 of Hernán and Robins [1].

    Reference: Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.
    Link: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    """

    @classmethod
    def setUpClass(cls) -> None:
        """ Get the data for examples in the second chapter: fake data from chapter 11 and NHEFS data from chapter 12
        onwards. NHEFS = National Health and Nutrition Examination Survey Data I Epidemiological Follow-up Study."""
        cls.nhefs_csv = pd.read_csv('https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv')
        A, Y = zip(*(
            (3, 21),
            (11, 54),
            (17, 33),
            (23, 101),
            (29, 85),
            (37, 65),
            (41, 157),
            (53, 120),
            (67, 111),
            (79, 200),
            (83, 140),
            (97, 220),
            (60, 230),
            (71, 217),
            (15, 11),
            (45, 190),
        ))
        cls.chapter_11_df = pd.DataFrame({'A': A, 'Y': Y, 'constant': np.ones(16)})

    def test_program_11_2(self):
        """Test whether our linear regression implementation produces the same results as program 11.2 (p. 141)."""
        df = self.chapter_11_df
        linear_regression_estimator = LinearRegressionEstimator(('A',), 100, 90, set(), ('Y',), df)
        intercept, coefficients = linear_regression_estimator._estimate_unit_effect()
        self.assertEqual(round(intercept[0] + 90*coefficients[0, 0], 1), 216.9)
        # Increasing A from 90 to 100 should be the same as 10 times the unit ATE
        self.assertEqual(10*coefficients[0, 0], linear_regression_estimator.estimate_average_treatment_effect())

    def test_program_11_3(self):
        """Test whether our linear regression implementation produces the same results as program 11.3 (p. 144)."""
        df = self.chapter_11_df
        linear_regression_estimator = LinearRegressionEstimator(('A',), 100, 90, set(), ('Y',), df)
        linear_regression_estimator.add_squared_term_to_df('A')
        intercept, coefficients = linear_regression_estimator._estimate_unit_effect()
        self.assertEqual(round(intercept[0] + 90*coefficients[0, 0] + 90*90*coefficients[0, 1], 1), 197.1)
        # Increasing A from 90 to 100 should be the same as 10 times the unit ATE
        self.assertEqual(round(10*coefficients[0, 0], 3),
                         round(linear_regression_estimator.estimate_average_treatment_effect(), 3))
