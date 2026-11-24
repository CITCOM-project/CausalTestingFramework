"""
This module tests the multinomial regression estimator against the scarf example from
https://investigate.ai/regression/multinomial-regression/.
(For binary categories, this should behave the same as for logistic regression)
"""

import pandas as pd

from causal_testing.estimation.multinomial_regression_estimator import MultinomialRegressionEstimator


def test_odds_ratio(scarf_df):
    """
    Test that the unit odds ratio for a binary outcome matches the logistic regression result.
    """
    multinomial_regression_estimator = MultinomialRegressionEstimator(
        treatment_variable="length_in",
        outcome_variable="completed",
        control_value=65,
        treatment_value=55,
        adjustment_set=set(),
    )
    effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio(scarf_df)
    assert round(effect_estimate.effect_estimate.iloc[0], 4) == 0.8948


def test_odds_ratio_category(scarf_df):
    """
    Test that a unit odds ratio is returned for each category of a categorical outcome.
    """
    multinomial_regression_estimator = MultinomialRegressionEstimator(
        treatment_variable="length_in",
        outcome_variable="color",
        control_value=65,
        treatment_value=55,
        adjustment_set=set(),
    )
    effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio(scarf_df)
    pd.testing.assert_series_equal(
        effect_estimate.effect_estimate.round(4).droplevel(1),
        pd.Series({"grey": 1.0072, "orange": 0.9668}).rename_axis("color"),
    )
