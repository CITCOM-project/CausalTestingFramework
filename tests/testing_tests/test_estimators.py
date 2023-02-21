import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.testing.estimators import (
    LinearRegressionEstimator,
    CausalForestEstimator,
    LogisticRegressionEstimator,
    InstrumentalVariableEstimator,
)
from causal_testing.specification.variable import Input


def plot_results_df(df):
    """A helper method to plot results dataframe for estimators, where the df parameter must have columns for the cate,
    ci_low, and ci_high.

    :param df: A dataframe containing the columns cate, ci_low, and ci_high, where each row is an observation.
    :return: Plot the treatment effect with confidence intervals for each observation.
    """

    df.sort_values("smokeintensity", inplace=True, ascending=True)
    df.reset_index(inplace=True, drop=True)
    plt.scatter(df["smokeintensity"], df["cate"], label="CATE", color="black")
    plt.fill_between(df["smokeintensity"], df["ci_low"], df["ci_high"], alpha=0.2)
    plt.ylabel("Weight Change (kg) caused by stopping smoking")
    plt.xlabel("Smoke intensity (cigarettes smoked per day)")
    plt.show()


def load_nhefs_df():
    """Get the NHEFS data from chapter 12 and put into a dataframe. NHEFS = National Health and Nutrition Examination
    Survey Data I Epidemiological Follow-up Study."""

    nhefs_df = pd.read_csv("tests/data/nhefs.csv")
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


class TestLogisticRegressionEstimator(unittest.TestCase):
    """Test the logistic regression estimator against the scarf example from
    https://investigate.ai/regression/logistic-regression/.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.DataFrame(
            [
                {"length_in": 55, "completed": 1},
                {"length_in": 55, "completed": 1},
                {"length_in": 55, "completed": 1},
                {"length_in": 60, "completed": 1},
                {"length_in": 60, "completed": 0},
                {"length_in": 70, "completed": 1},
                {"length_in": 70, "completed": 0},
                {"length_in": 82, "completed": 1},
                {"length_in": 82, "completed": 0},
                {"length_in": 82, "completed": 0},
                {"length_in": 82, "completed": 0},
            ]
        )

    def test_ate(self):
        df = self.scarf_df
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        ate, _ = logistic_regression_estimator.estimate_ate()
        self.assertEqual(round(ate, 4), -0.1987)

    def test_risk_ratio(self):
        df = self.scarf_df
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        rr, _ = logistic_regression_estimator.estimate_risk_ratio()
        self.assertEqual(round(rr, 4), 0.7664)

    def test_odds_ratio(self):
        df = self.scarf_df
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        odds = logistic_regression_estimator.estimate_unit_odds_ratio()
        self.assertEqual(round(odds, 4), 0.8948)


class TestInstrumentalVariableEstimator(unittest.TestCase):
    """
    Test the instrumental variable estimator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        Z = np.linspace(0, 10)
        X = 2 * Z
        Y = 2 * X
        cls.df = pd.DataFrame({"Z": Z, "X": X, "Y": Y})

    def test_estimate_coefficient(self):
        """
        Test we get the correct coefficient.
        """
        iv_estimator = InstrumentalVariableEstimator(
            treatment="X",
            treatment_value=None,
            control_value=None,
            adjustment_set=set(),
            outcome="Y",
            instrument="Z",
            df=self.df,
        )
        self.assertEqual(iv_estimator.estimate_coefficient(), 2)


class TestLinearRegressionEstimator(unittest.TestCase):
    """Test the linear regression estimator against the programming exercises in Section 2 of Hernán and Robins [1].

    Reference: Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.
    Link: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.nhefs_df = load_nhefs_df()
        cls.chapter_11_df = load_chapter_11_df()

    def test_program_11_2(self):
        """Test whether our linear regression implementation produces the same results as program 11.2 (p. 141)."""
        df = self.chapter_11_df
        linear_regression_estimator = LinearRegressionEstimator("treatments", 100, 90, set(), "outcomes", df)
        model = linear_regression_estimator._run_linear_regression()
        ate, _ = linear_regression_estimator.estimate_unit_ate()

        self.assertEqual(round(model.params["Intercept"] + 90 * model.params["treatments"], 1), 216.9)

        # Increasing treatments from 90 to 100 should be the same as 10 times the unit ATE
        self.assertEqual(round(10 * model.params["treatments"], 1), round(ate, 1))

    def test_program_11_3(self):
        """Test whether our linear regression implementation produces the same results as program 11.3 (p. 144)."""
        df = self.chapter_11_df.copy()
        linear_regression_estimator = LinearRegressionEstimator("treatments", 100, 90, set(), "outcomes", df)
        linear_regression_estimator.add_squared_term_to_df("treatments")
        model = linear_regression_estimator._run_linear_regression()
        ate, _ = linear_regression_estimator.estimate_unit_ate()
        self.assertEqual(
            round(
                model.params["Intercept"] + 90 * model.params["treatments"] + 90 * 90 * model.params["treatments^2"], 1
            ),
            197.1,
        )
        # Increasing treatments from 90 to 100 should be the same as 10 times the unit ATE
        self.assertEqual(round(10 * model.params["treatments"], 3), round(ate, 3))

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
        linear_regression_estimator = LinearRegressionEstimator("qsmk", 1, 0, covariates, "wt82_71", df)
        terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        terms_to_product = [("qsmk", "smokeintensity")]
        for term_to_square in terms_to_square:
            linear_regression_estimator.add_squared_term_to_df(term_to_square)
        for term_a, term_b in terms_to_product:
            linear_regression_estimator.add_product_term_to_df(term_a, term_b)

        model = linear_regression_estimator._run_linear_regression()
        self.assertEqual(round(model.params["qsmk"], 1), 2.6)
        self.assertEqual(round(model.params["qsmk*smokeintensity"], 2), 0.05)

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
        linear_regression_estimator = LinearRegressionEstimator("qsmk", 1, 0, covariates, "wt82_71", df)
        terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        for term_to_square in terms_to_square:
            linear_regression_estimator.add_squared_term_to_df(term_to_square)
        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_unit_ate()
        self.assertEqual(round(ate, 1), 3.5)
        self.assertEqual([round(ci_low, 1), round(ci_high, 1)], [2.6, 4.3])

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
        linear_regression_estimator = LinearRegressionEstimator("qsmk", 1, 0, covariates, "wt82_71", df)
        terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        for term_to_square in terms_to_square:
            linear_regression_estimator.add_squared_term_to_df(term_to_square)
        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_ate()
        self.assertEqual(round(ate, 1), 3.5)
        self.assertEqual([round(ci_low, 1), round(ci_high, 1)], [2.6, 4.3])

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
        linear_regression_estimator = LinearRegressionEstimator("qsmk", 1, 0, covariates, "wt82_71", df)
        terms_to_square = ["age", "wt71", "smokeintensity", "smokeyrs"]
        for term_to_square in terms_to_square:
            linear_regression_estimator.add_squared_term_to_df(term_to_square)
        ate, [ci_low, ci_high] = linear_regression_estimator.estimate_ate_calculated(
            {k: self.nhefs_df.mean()[k] for k in covariates}
        )
        self.assertEqual(round(ate, 1), 3.5)
        self.assertEqual([round(ci_low, 1), round(ci_high, 1)], [1.9, 5])


class TestCausalForestEstimator(unittest.TestCase):
    """Test the linear regression estimator against the programming exercises in Section 2 of Hernán and Robins [1].

    Reference: Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.
    Link: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.nhefs_df = load_nhefs_df()
        cls.chapter_11_df = load_chapter_11_df()

    def test_program_15_ate(self):
        """Test whether our causal forest implementation produces the similar ATE to program 15.1 (p. 163, 184)."""
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
        causal_forest = CausalForestEstimator(
            "qsmk", 1, 0, covariates, "wt82_71", df, {Input("smokeintensity", int): 40}
        )
        ate, _ = causal_forest.estimate_ate()
        self.assertGreater(round(ate, 1), 2.5)
        self.assertLess(round(ate, 1), 4.5)

    def test_program_15_cate(self):
        """Test whether our causal forest implementation produces the similar CATE to program 15.1 (p. 163, 184)."""
        df = self.nhefs_df
        smoking_intensity_5_and_40_df = df.loc[(df["smokeintensity"] == 5) | (df["smokeintensity"] == 40)]
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
        causal_forest = CausalForestEstimator(
            "qsmk", 1, 0, covariates, "wt82_71", smoking_intensity_5_and_40_df, {Input("smokeintensity", int): 40}
        )
        cates_df, _ = causal_forest.estimate_cates()
        self.assertGreater(cates_df["cate"].mean(), 0)
