"""
This module tests the logistic regression estimator against the scarf example from
https://investigate.ai/regression/logistic-regression/.
"""

from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator


def test_odds_ratio(scarf_df):
    """
    Test that the unit odds ratio matches the scarf example.
    """
    logistic_regression_estimator = LogisticRegressionEstimator(
        treatment_variable="length_in",
        outcome_variable="completed",
        control_value=65,
        treatment_value=55,
        adjustment_set=set(),
    )
    effect_estimate = logistic_regression_estimator.estimate_unit_odds_ratio(scarf_df)
    assert round(effect_estimate.effect_estimate.iloc[0], 4) == 0.8948
