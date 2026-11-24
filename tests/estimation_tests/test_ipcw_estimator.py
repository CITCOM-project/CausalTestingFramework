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


@pytest.fixture(name="df")
def _df():
    """
    Fresh copy of the temporal dataset for each test, with the status column added.
    """
    df = pd.read_csv("tests/resources/data/temporal_data.csv")
    df[STATUS_COLUMN] = df["outcome"] == 1
    return df


@pytest.fixture(name="make_estimation_model")
def _make_estimation_model(control_strategy, treatment_strategy):
    """
    Factory for estimators, using the default control and treatment strategies unless others are given.
    """

    def make(control_strategy=control_strategy, treatment_strategy=treatment_strategy):
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


@pytest.fixture(name="estimation_model")
def _estimation_model(make_estimation_model):
    """
    Fresh estimator for each test, using the default control and treatment strategies.
    """
    return make_estimation_model()


def test_estimate_hazard_ratio(estimation_model, df):
    effect_estimate = estimation_model.estimate_hazard_ratio(df)
    assert round(effect_estimate.effect_estimate["trtrand"], 3) == 1.351


def test_invalid_treatment_strategies(estimation_model, df):
    with pytest.raises(ValueError, match=re.escape("No individuals followed either strategy.")):
        estimation_model.preprocess_data(df.assign(t=(["1", "0"] * len(df))[: len(df)]))


def test_no_individual_began_control_strategy(estimation_model, control_strategy, df):
    with pytest.raises(ValueError, match=re.escape(f"No individuals began the control strategy {control_strategy}")):
        estimation_model.preprocess_data(df.assign(t=1))


def test_no_individual_began_treatment_strategy(estimation_model, treatment_strategy, df):
    with pytest.raises(
        ValueError, match=re.escape(f"No individuals began the treatment strategy {treatment_strategy}")
    ):
        estimation_model.preprocess_data(df.assign(t=0))


def test_preprocess_data_no_faults(estimation_model, df):
    with pytest.raises(ValueError, match=re.escape("No safe runs with outcome. Cannot perform estimation.")):
        estimation_model.preprocess_data(df.assign(**{STATUS_COLUMN: True}))


def test_preprocess_data_only_faults(estimation_model, df):
    with pytest.raises(ValueError, match=re.escape("No faults with outcome. Cannot perform estimation.")):
        estimation_model.preprocess_data(df.assign(**{STATUS_COLUMN: False}))


def test_preprocess_data_missing_control(make_estimation_model, control_strategy, df):
    estimation_model = make_estimation_model(control_strategy=control_strategy + [(0, "invalid", 1)])
    with pytest.raises(ValueError, match=re.escape("Missing data for control strategy.")):
        estimation_model.preprocess_data(df.assign(**{STATUS_COLUMN: True}))


def test_preprocess_data_missing_treatment(make_estimation_model, treatment_strategy, df):
    estimation_model = make_estimation_model(treatment_strategy=treatment_strategy + [(0, "invalid", 1)])
    with pytest.raises(ValueError, match=re.escape("Missing data for treatment strategy.")):
        estimation_model.preprocess_data(df.assign(**{STATUS_COLUMN: True}))
