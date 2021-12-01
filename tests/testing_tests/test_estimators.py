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
        cls.nhefs_df = pd.read_csv('https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv')
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
        cls.nhefs_df['one'] = 1
        cls.nhefs_df['zero'] = 0
        edu_dummies = pd.get_dummies(cls.nhefs_df.education, prefix='edu')
        exercise_dummies = pd.get_dummies(cls.nhefs_df.exercise, prefix='exercise')
        active_dummies = pd.get_dummies(cls.nhefs_df.active, prefix='active')
        cls.nhefs_df = pd.concat([cls.nhefs_df, edu_dummies, exercise_dummies, active_dummies], axis=1)

    def test_program_11_2(self):
        """Test whether our linear regression implementation produces the same results as program 11.2 (p. 141)."""
        df = self.chapter_11_df
        linear_regression_estimator = LinearRegressionEstimator(('A',), 100, 90, {'constant'}, ('Y',), df)
        model = linear_regression_estimator._run_linear_regression()
        ate, _ = linear_regression_estimator.estimate_unit_ate()

        self.assertEqual(round(model.params['constant'] + 90*model.params['A'], 1), 216.9)

        # Increasing A from 90 to 100 should be the same as 10 times the unit ATE
        self.assertEqual(round(10*model.params['A'], 1), round(ate, 1))

    def test_program_11_3(self):
        """Test whether our linear regression implementation produces the same results as program 11.3 (p. 144)."""
        df = self.chapter_11_df.copy()
        linear_regression_estimator = LinearRegressionEstimator(('A',), 100, 90, {'constant'}, ('Y',), df)
        linear_regression_estimator.add_squared_term_to_df('A')
        model = linear_regression_estimator._run_linear_regression()
        ate, _ = linear_regression_estimator.estimate_unit_ate()
        self.assertEqual(round(model.params['constant'] + 90*model.params['A'] + 90*90*model.params['A^2'], 1), 197.1)
        # Increasing A from 90 to 100 should be the same as 10 times the unit ATE
        self.assertEqual(round(10*model.params['A'], 3), round(ate, 3))

    def test_program_15_1A(self):
        """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)."""
        df = self.nhefs_df
        covariates = {'sex', 'race', 'age', 'edu_2', 'edu_3', 'edu_4', 'edu_5', 'exercise_1', 'exercise_2',
                      'active_1', 'active_2', 'wt71', 'smokeintensity', 'smokeyrs'}
        linear_regression_estimator = LinearRegressionEstimator(('qsmk',), 1, 0, covariates, ('wt82_71',), df)
        terms_to_square = ['age', 'wt71', 'smokeintensity', 'smokeyrs']
        terms_to_product = [('qsmk', 'smokeintensity')]
        for term_to_square in terms_to_square:
            linear_regression_estimator.add_squared_term_to_df(term_to_square)
        for term_a, term_b in terms_to_product:
            linear_regression_estimator.add_product_term_to_df(term_a, term_b)

        model = linear_regression_estimator._run_linear_regression()
        self.assertEqual(round(model.params['qsmk'], 1), 2.6)
        self.assertEqual(round(model.params['qsmk*smokeintensity'], 2), 0.05)

    def test_program_15_no_interaction(self):
        """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
        without product parameter. """
        df = self.nhefs_df
        covariates = {'sex', 'race', 'age', 'edu_2', 'edu_3', 'edu_4', 'edu_5', 'exercise_1', 'exercise_2',
                      'active_1', 'active_2', 'wt71', 'smokeintensity', 'smokeyrs'}
        linear_regression_estimator = LinearRegressionEstimator(('qsmk',), 1, 0, covariates, ('wt82_71',),
                                                                df)
        terms_to_square = ['age', 'wt71', 'smokeintensity', 'smokeyrs']
        for term_to_square in terms_to_square:
            linear_regression_estimator.add_squared_term_to_df(term_to_square)
        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_unit_ate()
        self.assertEqual(round(ate, 1), 3.5)
        self.assertEqual([round(ci_low, 1), round(ci_high, 1)], [2.6, 4.3])

