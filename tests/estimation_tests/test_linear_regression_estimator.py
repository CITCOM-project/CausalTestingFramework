"""
This module tests the linear regression estimator against the programming exercises in Section 2 of Hernán and Robins [1].

Reference: Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.
Link: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
"""

import re

import numpy as np
import pandas as pd
import pytest

from causal_testing.estimation.genetic_programming_regression_fitter import reciprocal
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.utils.validation import CausalValidator

NO_INTERACTION_FORMULA = (
    "wt82_71 ~ qsmk + age + I(age ** 2) + wt71 + I(wt71 ** 2) + smokeintensity + I(smokeintensity ** 2) + smokeyrs "
    "+ I(smokeyrs ** 2)"
)


@pytest.fixture(name="nhefs_df")
def _nhefs_df():
    """Get the NHEFS data from chapter 12 and put into a dataframe. NHEFS = National Health and Nutrition Examination
    Survey Data I Epidemiological Follow-up Study."""

    nhefs_df = pd.read_csv("tests/resources/data/nhefs.csv").assign(one=1, zero=0)
    edu_dummies = pd.get_dummies(nhefs_df.education, prefix="edu")
    exercise_dummies = pd.get_dummies(nhefs_df.exercise, prefix="exercise")
    active_dummies = pd.get_dummies(nhefs_df.active, prefix="active")
    nhefs_df = pd.concat([nhefs_df, edu_dummies, exercise_dummies, active_dummies], axis=1)
    return nhefs_df


@pytest.fixture(name="chapter_11_df")
def _chapter_11_df():
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


@pytest.fixture(name="scarf_df", scope="module")
def _scarf_df():
    """
    Scarf data, loaded once for the module.
    """
    return pd.read_csv("tests/resources/data/scarf_data.csv")


@pytest.fixture(name="interaction_df", scope="module")
def _interaction_df():
    """
    Synthetic data with an interaction term: Y = 2X1 - 3X2 + 2*X1*X2 + 10
    """
    df = pd.DataFrame({"X1": np.random.uniform(-1000, 1000, 1000), "X2": np.random.uniform(-1000, 1000, 1000)})
    df["Y"] = 2 * df["X1"] - 3 * df["X2"] + 2 * df["X1"] * df["X2"] + 10
    return df


@pytest.mark.parametrize(
    "formula, expected_adjustment_set",
    [
        # Test that the adjustment set can be extracted from a complex formula without ignoring variables that share
        # their names with key terms like C (for categoricals) or the treatment variable.
        (
            "price ~ depth:color + C(cut, Treatment('Good')) + cr(np.minimum(C, 0.8), df=5, constraints='center')",
            {"C", "color", "depth"},
        ),
        # Test that key terms are not included in the adjustment set.
        (
            "price ~ I(color ** 2) + C(cut, Treatment('Good')) + Q(np.minimum(caret, 0.8))",
            {"caret", "color"},
        ),
    ],
    ids=["complex_formula", "no_c"],
)
def test_complex_formula_adjustment_set(formula, expected_adjustment_set):
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="cut", outcome_variable="price", formula=formula
    )
    assert linear_regression_estimator.adjustment_set == expected_adjustment_set


def test_complex_formula_adjustment_set_no_dependent():
    """
    Test that the outcome variable is prepended to the formula if no dependent variable is specified.
    """
    formula = "I(color ** 2) + C(cut, Treatment('Good')) + Q(np.minimum(caret, 0.8))"
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="cut", outcome_variable="price", formula=formula
    )
    assert linear_regression_estimator.formula == f"price ~ {formula}"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        # Test that an error is thrown if there is a mismatch between the outcome variable and the dependent variable.
        (
            {"formula": "incorrect ~ I(color ** 2) + C(cut, Treatment('Good')) + Q(np.minimum(caret, 0.8))"},
            "Left hand side of formula incorrect ~ I(color ** 2) + C(cut, Treatment('Good')) + "
            "Q(np.minimum(caret, 0.8)) does not match the specified outcome_variable price.",
        ),
        # Test that an error is thrown when the adjustment set implied by the formula does not match the one specified.
        (
            {
                "formula": "price ~ I(color ** 2) + C(cut, Treatment('Good')) + Q(np.minimum(caret, 0.8))",
                "adjustment_set": set(),
            },
            "Specified formula price ~ I(color ** 2) + C(cut, Treatment('Good')) + Q(np.minimum(caret, 0.8)) "
            "does not match specified adjustment set set()",
        ),
        # Test that an error is thrown when the adjustment configuration does not match the adjustment set.
        (
            {"adjustment_set": {"caret"}, "adjustment_config": {"color": "green"}},
            "Specified configuration for variables ['color'] which are not in the adjustment set {'caret'}.",
        ),
        # Test that an error is thrown when neither the formula nor the adjustment set is specified.
        ({}, "Please specify either a formula or an adjustment set."),
    ],
    ids=["wrong_dependent", "adjustment_set_mismatch", "adjustment_config_mismatch", "no_formula_or_adjustment_set"],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        LinearRegressionEstimator(treatment_variable="cut", outcome_variable="price", **kwargs)


def test_formula_from_adjustment_set():
    """
    Test that the correct formula is built from the adjustment set.
    """
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="cut", outcome_variable="price", adjustment_set={"caret", "color"}
    )
    assert linear_regression_estimator.formula == "price ~ cut + caret + color"


def test_linear_regression_categorical_ate(scarf_df):
    df = scarf_df.copy()
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="color", outcome_variable="completed", adjustment_set=set()
    )
    effect_estimate = linear_regression_estimator.estimate_coefficient(df)
    assert all(ci_low < 0 < ci_high for ci_low, ci_high in zip(effect_estimate.ci_low, effect_estimate.ci_high))


@pytest.mark.parametrize(
    "estimator_kwargs, decimals",
    [
        # Program 11.2 (p. 141)
        ({"adjustment_set": set()}, 1),
        # Program 11.3 (p. 144)
        ({"formula": "outcomes ~ treatments + I(treatments ** 2)"}, 3),
    ],
    ids=["program_11_2", "program_11_3"],
)
def test_program_11(chapter_11_df, estimator_kwargs, decimals):
    """Test whether our linear regression implementation produces the same results as programs 11.2 and 11.3."""
    df = chapter_11_df.copy()
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="treatments", outcome_variable="outcomes", **estimator_kwargs
    )
    effect_estimate = linear_regression_estimator.estimate_coefficient(df)
    # Increasing treatments from 90 to 100 should be the same as 10 times the unit ATE
    assert all(
        round(effect_estimate.effect_estimate["treatments"], decimals) == round(ate_single, decimals)
        for ate_single in effect_estimate.effect_estimate
    )


def test_program_15_1A(nhefs_df):
    """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)."""
    df = nhefs_df
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
        treatment_variable="qsmk",
        outcome_variable="wt82_71",
        treatment_value=1,
        control_value=0,
        formula=f"""wt82_71 ~ qsmk +
                         {'+'.join(sorted(list(covariates)))} +
                         I(age ** 2) +
                         I(wt71 ** 2) +
                         I(smokeintensity ** 2) +
                         I(smokeyrs ** 2) +
                         (qsmk * smokeintensity)""",
    )

    effect_estimate = linear_regression_estimator.estimate_coefficient(df)
    assert round(effect_estimate.effect_estimate["qsmk"], 1) == 2.6


def test_program_15_no_interaction(nhefs_df):
    """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
    without product parameter."""
    df = nhefs_df
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="qsmk",
        outcome_variable="wt82_71",
        treatment_value=1,
        control_value=0,
        formula=NO_INTERACTION_FORMULA,
    )
    effect_estimate = linear_regression_estimator.estimate_coefficient(df)

    assert round(effect_estimate.effect_estimate.iloc[0], 1) == 3.5
    assert round(effect_estimate.ci_low.iloc[0], 1) == 2.6
    assert round(effect_estimate.ci_high.iloc[0], 1) == 4.3


def test_program_15_no_interaction_ate(nhefs_df):
    """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
    without product parameter."""
    df = nhefs_df
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="qsmk",
        outcome_variable="wt82_71",
        treatment_value=1,
        control_value=0,
        formula=NO_INTERACTION_FORMULA,
    )
    effect_estimate = linear_regression_estimator.estimate_ate(df)
    assert round(effect_estimate.effect_estimate[0], 1) == 3.5
    assert [round(effect_estimate.ci_low[0], 1), round(effect_estimate.ci_high[0], 1)] == [2.6, 4.3]


def test_program_15_no_interaction_ate_calculated(nhefs_df):
    """Test whether our linear regression implementation produces the same results as program 15.1 (p. 163, 184)
    without product parameter."""
    covariates = {"age", "wt71", "smokeintensity", "smokeyrs"}
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="qsmk",
        outcome_variable="wt82_71",
        treatment_value=1,
        control_value=0,
        formula=NO_INTERACTION_FORMULA,
        adjustment_config={k: nhefs_df.mean()[k] for k in covariates},
    )

    effect_estimate = linear_regression_estimator.estimate_ate_calculated(df=nhefs_df)
    assert round(effect_estimate.effect_estimate[0], 1) == 3.5
    assert [round(effect_estimate.ci_low[0], 1), round(effect_estimate.ci_high[0], 1)] == [1.9, 5]


def test_program_11_2_with_robustness_validation(chapter_11_df):
    """Test whether our linear regression estimator, as used in test_program_11_2 can correctly estimate robustness."""
    df = chapter_11_df.copy()
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="treatments",
        outcome_variable="outcomes",
        treatment_value=100,
        control_value=90,
        adjustment_set=set(),
    )

    cv = CausalValidator()
    assert round(cv.estimate_robustness(linear_regression_estimator.fit_model(df))["treatments"], 4) == 0.7353


def test_to_dict():
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="X",
        outcome_variable="Y",
        control_value=0,
        treatment_value=1,
        adjustment_set={"Z"},
        adjustment_config={"Z": 1},
    )
    assert linear_regression_estimator.to_dict() == {
        "name": "LinearRegressionEstimator",
        "treatment_variable": "X",
        "outcome_variable": "Y",
        "alpha": 0.05,
        "formula": "Y ~ X + Z",
        "adjustment_set": ["Z"],
        "adjustment_config": {"Z": 1},
        "control_value": 0,
        "treatment_value": 1,
    }


@pytest.mark.parametrize(
    "x_values, y_function, gp_kwargs, expected_formula, expected_estimate",
    [
        (
            np.arange(10).astype(float),
            lambda x: 1 / (x + 1),
            {"seeds": ["reciprocal(add(X, 1))"], "extra_operators": [(reciprocal, 1)]},
            "Y ~ I(1/(X + 1)) - 1",
            0.50,
        ),
        (
            np.arange(10),
            lambda x: 2 * (x**2),
            {"seed": 1, "max_order": 2, "seeds": ["mul(2, power_2(X))"], "extra_operators": []},
            "Y ~ I(2*X**2) - 1",
            -2.00,
        ),
    ],
    ids=["reciprocal", "power"],
)
def test_gp(x_values, y_function, gp_kwargs, expected_formula, expected_estimate):
    df = pd.DataFrame()
    df["X"] = x_values
    df["Y"] = y_function(df["X"])
    linear_regression_estimator = LinearRegressionEstimator(
        treatment_variable="X",
        outcome_variable="Y",
        treatment_value=0,
        control_value=1,
        adjustment_set=set(),
    )
    linear_regression_estimator.gp_formula(df=df, **gp_kwargs)
    assert linear_regression_estimator.formula == expected_formula
    effect_estimate = linear_regression_estimator.estimate_ate_calculated(df)
    assert round(effect_estimate.effect_estimate[0], 2) == expected_estimate
    assert round(effect_estimate.ci_low[0], 2) == expected_estimate
    assert round(effect_estimate.ci_high[0], 2) == expected_estimate


def test_X1_effect(interaction_df):
    """When we fix the value of X2 to 0, the effect of X1 on Y should become ~2 (because X2 terms are cancelled)."""
    lr_model = LinearRegressionEstimator(
        treatment_variable="X1",
        outcome_variable="Y",
        treatment_value=1,
        control_value=0,
        adjustment_set={"X2"},
        adjustment_config={"X2": 0},
        formula="Y ~ X1 + X2 + (X1 * X2)",
    )
    effect_estimate = lr_model.estimate_ate(interaction_df)
    assert effect_estimate.effect_estimate[0] == pytest.approx(2.0, abs=5e-8)


def test_categorical_confidence_intervals(scarf_df):
    lr_model = LinearRegressionEstimator(treatment_variable="color", outcome_variable="length_in", adjustment_set=set())
    effect_estimate = lr_model.estimate_coefficient(scarf_df)

    # The precise values don't really matter. This test is primarily intended to make sure the return type is correct.
    assert effect_estimate.effect_estimate.round(2).equals(pd.Series({"color[T.grey]": 0.92, "color[T.orange]": -4.25}))
    assert effect_estimate.ci_low.round(2).equals(pd.Series({"color[T.grey]": -22.12, "color[T.orange]": -25.58}))
    assert effect_estimate.ci_high.round(2).equals(pd.Series({"color[T.grey]": 23.95, "color[T.orange]": 17.08}))


def test_program_11_3_linear_regression(chapter_11_df):
    """Test whether the cublic_spline regression implementation produces the same results as program 11.3 (p. 162).
    https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2023/10/hernanrobins_WhatIf_30sep23.pdf
    Slightly modified as Hernan et al. use linear regression for this example.
    """

    cublic_spline_estimator = LinearRegressionEstimator(
        treatment_variable="treatments",
        outcome_variable="outcomes",
        treatment_value=1,
        control_value=0,
        formula="outcomes ~ cr(treatments, df=3)",
    )

    ate_1 = cublic_spline_estimator.estimate_ate_calculated(chapter_11_df).effect_estimate

    cublic_spline_estimator.treatment_value = 2
    ate_2 = cublic_spline_estimator.estimate_ate_calculated(chapter_11_df).effect_estimate

    # Doubling the treatemebnt value should roughly but not exactly double the ATE
    assert ate_1[0] * 2 != ate_2[0]
    assert ate_1[0] * 2 == pytest.approx(ate_2[0], abs=5e-8)
