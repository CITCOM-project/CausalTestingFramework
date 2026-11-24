"""
This module tests the IPCW estimator.
"""

import re

import pandas as pd
import pytest

from causal_testing.estimation.ipcw_estimator import IPCWEstimator

STATUS_COLUMN = "ok"
TIMESTEPS_PER_OBSERVATION = 1
FIT_BL_SWITCH_FORMULA = "xo_t_do ~ time"


@pytest.fixture(name="control_strategy")
def _control_strategy():
    """
    Control strategy: t is held at 0 for timesteps 1 to 3.
    """
    return [[t, "t", 0] for t in range(1, 4, TIMESTEPS_PER_OBSERVATION)]


@pytest.fixture(name="treatment_strategy")
def _treatment_strategy():
    """
    Treatment strategy: t is held at 1 for timesteps 1 to 3.
    """
    return [[t, "t", 1] for t in range(1, 4, TIMESTEPS_PER_OBSERVATION)]


@pytest.fixture(name="temporal_data")
def _temporal_data():
    """
    Fresh copy of the temporal dataset for each test, with the status column added.
    """
    df = pd.read_csv("tests/resources/data/temporal_data.csv")
    df[STATUS_COLUMN] = df["outcome"] == 1
    return df


@pytest.fixture(name="estimation_model")
def _estimation_model(control_strategy, treatment_strategy):
    """
    Factory for estimators, using the default control and treatment strategies unless others are given.
    """

    def make(control_strategy=control_strategy, treatment_strategy=treatment_strategy):
        """
        Create an estimator with the given control and treatment strategies.
        """
        return IPCWEstimator(
            timesteps_per_observation=TIMESTEPS_PER_OBSERVATION,
            control_strategy=control_strategy,
            treatment_strategy=treatment_strategy,
            outcome_variable="outcome",
            status_column=STATUS_COLUMN,
            fit_bl_switch_formula=FIT_BL_SWITCH_FORMULA,
            fit_bltd_switch_formula=FIT_BL_SWITCH_FORMULA,
            eligibility=None,
        )

    return make


def test_estimate_hazard_ratio(estimation_model, temporal_data):
    """
    Test that the hazard ratio is estimated correctly.
    """
    effect_estimate = estimation_model().estimate_hazard_ratio(temporal_data)
    assert round(effect_estimate.effect_estimate["trtrand"], 3) == 1.351


@pytest.mark.parametrize(
    "estimation_model_kwargs, assignments, error_message",
    [
        # No individuals follow either strategy.
        (
            {},
            {"t": lambda temporal_data: (["1", "0"] * len(temporal_data))[: len(temporal_data)]},
            "No individuals followed either strategy.",
        ),
        # No individuals began the control strategy.
        ({}, {"t": 1}, "No individuals began the control strategy {control_strategy}"),
        # No individuals began the treatment strategy.
        ({}, {"t": 0}, "No individuals began the treatment strategy {treatment_strategy}"),
        # There are no faults in the data.
        ({}, {STATUS_COLUMN: True}, "No faults with outcome. Cannot perform estimation."),
        # There are only faults in the data.
        ({}, {STATUS_COLUMN: False}, "No safe runs with outcome. Cannot perform estimation."),
        # The data has no column for a variable in the control strategy.
        ({"control": [(0, "invalid", 1)]}, {STATUS_COLUMN: True}, "Missing data for control strategy."),
        # The data has no column for a variable in the treatment strategy.
        ({"treatment": [(0, "invalid", 1)]}, {STATUS_COLUMN: True}, "Missing data for treatment strategy."),
    ],
    ids=[
        "invalid_treatment_strategies",
        "no_individual_began_control_strategy",
        "no_individual_began_treatment_strategy",
        "no_faults",
        "only_faults",
        "missing_control",
        "missing_treatment",
    ],
)
def test_preprocess_data_invalid(
    estimation_model,
    control_strategy,
    treatment_strategy,
    temporal_data,
    estimation_model_kwargs,
    assignments,
    error_message,
):  # pylint: disable=too-many-positional-arguments
    """
    Test that preprocess_data raises the correct ValueError for invalid strategies or data.
    """
    expected = error_message.format(control_strategy=control_strategy, treatment_strategy=treatment_strategy)
    with pytest.raises(ValueError, match=re.escape(expected)):
        estimation_model(
            control_strategy=control_strategy + estimation_model_kwargs.get("control", []),
            treatment_strategy=treatment_strategy + estimation_model_kwargs.get("treatment", []),
        ).preprocess_data(temporal_data.assign(**assignments))
