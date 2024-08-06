import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.variable import Input
from causal_testing.utils.validation import CausalValidator
from causal_testing.specification.capabilities import TreatmentSequence

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator


def load_nhefs_df():
    """Get the NHEFS data from chapter 12 and put into a dataframe. NHEFS = National Health and Nutrition Examination
    Survey Data I Epidemiological Follow-up Study."""

    nhefs_df = pd.read_csv("tests/resources/data/nhefs.csv")
    nhefs_df["one"] = 1
    nhefs_df["zero"] = 0
    edu_dummies = pd.get_dummies(nhefs_df.education, prefix="edu")
    exercise_dummies = pd.get_dummies(nhefs_df.exercise, prefix="exercise")
    active_dummies = pd.get_dummies(nhefs_df.active, prefix="active")
    nhefs_df = pd.concat([nhefs_df, edu_dummies, exercise_dummies, active_dummies], axis=1)
    return nhefs_df


def load_chapter_11_df():
    """Get the data from chapter 11 and put into a dataframe."""

    treatments, outcomes = zip(
        *(
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
        )
    )
    chapter_11_df = pd.DataFrame({"treatments": treatments, "outcomes": outcomes, "constant": np.ones(16)})
    return chapter_11_df


class TestLinearRegressionEstimator(unittest.TestCase):
    """Test the linear regression estimator against the programming exercises in Section 2 of Hernán and Robins [1].

    Reference: Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.
    Link: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.nhefs_df = load_nhefs_df()
        cls.chapter_11_df = load_chapter_11_df()
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_query(self):
        df = self.nhefs_df
        linear_regression_estimator = LinearRegressionEstimator(
            "treatments", None, None, set(), "outcomes", df, query="sex==1"
        )
        self.assertTrue(linear_regression_estimator.df.sex.all())

    def test_linear_regression_categorical_ate(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LinearRegressionEstimator("color", None, None, set(), "completed", df)
        ate, confidence = logistic_regression_estimator.estimate_coefficient()
        self.assertTrue(all([ci_low < 0 < ci_high for ci_low, ci_high in zip(confidence[0], confidence[1])]))

    def test_program_11_2(self):
        """Test whether our linear regression implementation produces the same results as program 11.2 (p. 141)."""
        df = self.chapter_11_df
        linear_regression_estimator = LinearRegressionEstimator("treatments", None, None, set(), "outcomes", df)
        model = linear_regression_estimator._run_linear_regression()
        ate, _ = linear_regression_estimator.estimate_coefficient()

        self.assertEqual(round(model.params["Intercept"] + 90 * model.params["treatments"], 1), 216.9)

        # Increasing treatments from 90 to 100 should be the same as 10 times the unit ATE
        self.assertTrue(all(round(model.params["treatments"], 1) == round(ate_single, 1) for ate_single in ate))

    def test_program_11_3(self):
        """Test whether our linear regression implementation produces the same results as program 11.3 (p. 144)."""
        df = self.chapter_11_df.copy()
        linear_regression_estimator = LinearRegressionEstimator(
            "treatments", None, None, set(), "outcomes", df, formula="outcomes ~ treatments + np.power(treatments, 2)"
        )
        model = linear_regression_estimator._run_linear_regression()
        ate, _ = linear_regression_estimator.estimate_coefficient()
        self.assertEqual(
            round(
                model.params["Intercept"]
                + 90 * model.params["treatments"]
                + 90 * 90 * model.params["np.power(treatments, 2)"],
                1,
            ),
            197.1,
        )
        # Increasing treatments from 90 to 100 should be the same as 10 times the unit ATE
        self.assertTrue(all(round(model.params["treatments"], 3) == round(ate_single, 3) for ate_single in ate))

    def test_program_15_1A(self):
        """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)."""
        df = self.nhefs_df
        covariates = {
            "sex",
            "race",
            "age",
            "edu_2",
            "edu_3",
            "edu_4",
            "edu_5",
            "exercise_1",
            "exercise_2",
            "active_1",
            "active_2",
            "wt71",
            "smokeintensity",
            "smokeyrs",
        }
        linear_regression_estimator = LinearRegressionEstimator(
            "qsmk",
            1,
            0,
            covariates,
            "wt82_71",
            df,
            formula=f"""wt82_71 ~ qsmk +
                             {'+'.join(sorted(list(covariates)))} +
                             np.power(age, 2) +
                             np.power(wt71, 2) +
                             np.power(smokeintensity, 2) +
                             np.power(smokeyrs, 2) +
                             (qsmk * smokeintensity)""",
        )
        # terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        # terms_to_product = [("qsmk", "smokeintensity")]
        # for term_to_square in terms_to_square:
        # for term_a, term_b in terms_to_product:
        #     linear_regression_estimator.add_product_term_to_df(term_a, term_b)

        model = linear_regression_estimator._run_linear_regression()
        self.assertEqual(round(model.params["qsmk"], 1), 2.6)
        self.assertEqual(round(model.params["qsmk:smokeintensity"], 2), 0.05)

    def test_program_15_no_interaction(self):
        """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
        without product parameter."""
        df = self.nhefs_df
        covariates = {
            "sex",
            "race",
            "age",
            "edu_2",
            "edu_3",
            "edu_4",
            "edu_5",
            "exercise_1",
            "exercise_2",
            "active_1",
            "active_2",
            "wt71",
            "smokeintensity",
            "smokeyrs",
        }
        linear_regression_estimator = LinearRegressionEstimator(
            "qsmk",
            1,
            0,
            covariates,
            "wt82_71",
            df,
            formula="wt82_71 ~ qsmk + age + np.power(age, 2) + wt71 + np.power(wt71, 2) + smokeintensity + np.power(smokeintensity, 2) + smokeyrs + np.power(smokeyrs, 2)",
        )
        # terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        # for term_to_square in terms_to_square:
        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_coefficient()

        self.assertEqual(round(ate[0], 1), 3.5)
        self.assertEqual([round(ci_low[0], 1), round(ci_high[0], 1)], [2.6, 4.3])

    def test_program_15_no_interaction_ate(self):
        """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
        without product parameter."""
        df = self.nhefs_df
        covariates = {
            "sex",
            "race",
            "age",
            "edu_2",
            "edu_3",
            "edu_4",
            "edu_5",
            "exercise_1",
            "exercise_2",
            "active_1",
            "active_2",
            "wt71",
            "smokeintensity",
            "smokeyrs",
        }
        linear_regression_estimator = LinearRegressionEstimator(
            "qsmk",
            1,
            0,
            covariates,
            "wt82_71",
            df,
            formula="wt82_71 ~ qsmk + age + np.power(age, 2) + wt71 + np.power(wt71, 2) + smokeintensity + np.power(smokeintensity, 2) + smokeyrs + np.power(smokeyrs, 2)",
        )
        # terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        # for term_to_square in terms_to_square:
        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_ate()
        self.assertEqual(round(ate[0], 1), 3.5)
        self.assertEqual([round(ci_low[0], 1), round(ci_high[0], 1)], [2.6, 4.3])

    def test_program_15_no_interaction_ate_calculated(self):
        """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
        without product parameter."""
        df = self.nhefs_df
        covariates = {
            "sex",
            "race",
            "age",
            "edu_2",
            "edu_3",
            "edu_4",
            "edu_5",
            "exercise_1",
            "exercise_2",
            "active_1",
            "active_2",
            "wt71",
            "smokeintensity",
            "smokeyrs",
        }
        linear_regression_estimator = LinearRegressionEstimator(
            "qsmk",
            1,
            0,
            covariates,
            "wt82_71",
            df,
            formula="wt82_71 ~ qsmk + age + np.power(age, 2) + wt71 + np.power(wt71, 2) + smokeintensity + np.power(smokeintensity, 2) + smokeyrs + np.power(smokeyrs, 2)",
        )
        # terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        # for term_to_square in terms_to_square:

        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_ate_calculated(
            adjustment_config={k: self.nhefs_df.mean()[k] for k in covariates}
        )
        self.assertEqual(round(ate[0], 1), 3.5)
        self.assertEqual([round(ci_low[0], 1), round(ci_high[0], 1)], [1.9, 5])

    def test_program_11_2_with_robustness_validation(self):
        """Test whether our linear regression estimator, as used in test_program_11_2 can correctly estimate robustness."""
        df = self.chapter_11_df.copy()
        linear_regression_estimator = LinearRegressionEstimator("treatments", 100, 90, set(), "outcomes", df)
        model = linear_regression_estimator._run_linear_regression()

        cv = CausalValidator()
        self.assertEqual(round(cv.estimate_robustness(model)["treatments"], 4), 0.7353)

    def test_gp(self):
        df = pd.DataFrame()
        df["X"] = np.arange(10)
        df["Y"] = 1 / (df["X"] + 1)
        linear_regression_estimator = LinearRegressionEstimator("X", 0, 1, set(), "Y", df.astype(float))
        linear_regression_estimator.gp_formula(seed=1)
        self.assertEqual(
            linear_regression_estimator.formula,
            "Y ~ I((2.606801258739728e-17*X + 0.626132756132756)/(0.6261327561327561*X + 0.626132756132756)) - 1",
        )
        ate, (ci_low, ci_high) = linear_regression_estimator.estimate_ate_calculated()
        self.assertEqual(round(ate[0], 2), 0.50)
        self.assertEqual(round(ci_low[0], 2), 0.50)
        self.assertEqual(round(ci_high[0], 2), 0.50)


class TestLinearRegressionInteraction(unittest.TestCase):
    """Test linear regression for estimating effects involving interaction."""

    @classmethod
    def setUpClass(cls) -> None:
        # Y = 2X1 - 3X2 + 2*X1*X2 + 10
        df = pd.DataFrame({"X1": np.random.uniform(-1000, 1000, 1000), "X2": np.random.uniform(-1000, 1000, 1000)})
        df["Y"] = 2 * df["X1"] - 3 * df["X2"] + 2 * df["X1"] * df["X2"] + 10
        cls.df = df
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_X1_effect(self):
        """When we fix the value of X2 to 0, the effect of X1 on Y should become ~2 (because X2 terms are cancelled)."""
        lr_model = LinearRegressionEstimator(
            "X1", 1, 0, {"X2"}, "Y", effect_modifiers={"x2": 0}, formula="Y ~ X1 + X2 + (X1 * X2)", df=self.df
        )
        test_results = lr_model.estimate_ate()
        ate = test_results[0][0]
        self.assertAlmostEqual(ate, 2.0)

    def test_categorical_confidence_intervals(self):
        lr_model = LinearRegressionEstimator(
            treatment="color",
            control_value=None,
            treatment_value=None,
            adjustment_set={},
            outcome="length_in",
            df=self.scarf_df,
        )
        coefficients, [ci_low, ci_high] = lr_model.estimate_coefficient()

        # The precise values don't really matter. This test is primarily intended to make sure the return type is correct.
        self.assertTrue(coefficients.round(2).equals(pd.Series({"color[T.grey]": 0.92, "color[T.orange]": -4.25})))
        self.assertTrue(ci_low.round(2).equals(pd.Series({"color[T.grey]": -22.12, "color[T.orange]": -25.58})))
        self.assertTrue(ci_high.round(2).equals(pd.Series({"color[T.grey]": 23.95, "color[T.orange]": 17.08})))
