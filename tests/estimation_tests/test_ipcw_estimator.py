import unittest

import pandas as pd

from causal_testing.estimation.ipcw_estimator import IPCWEstimator


class TestIPCWEstimator(unittest.TestCase):
    """
    Test the IPCW estimator class
    """

    def setUp(self) -> None:
        self.status_column = "ok"
        self.timesteps_per_observation = 1
        self.control_strategy = [[t, "t", 0] for t in range(1, 4, self.timesteps_per_observation)]
        self.treatment_strategy = [[t, "t", 1] for t in range(1, 4, self.timesteps_per_observation)]
        self.fit_bl_switch_formula = "xo_t_do ~ time"
        self.df = pd.read_csv("tests/resources/data/temporal_data.csv")
        self.df[self.status_column] = self.df["outcome"] == 1

    def test_estimate_hazard_ratio(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        effect_estimate = estimation_model.estimate_hazard_ratio(self.df)
        self.assertEqual(round(effect_estimate.effect_estimate["trtrand"], 3), 1.351)

    def test_invalid_treatment_strategies(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaises(ValueError):
            estimation_model.preprocess_data(self.df.assign(t=(["1", "0"] * len(self.df))[: len(self.df)]))

    def test_invalid_fault_t_do(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaises(ValueError):
            estimation_model.estimate_hazard_ratio(self.df.loc[~self.df["ok"]])

    def test_no_individual_began_control_strategy(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaisesRegex(ValueError, "No individuals began the control strategy .*"):
            estimation_model.preprocess_data(self.df.assign(t=1))

    def test_no_individual_began_treatment_strategy(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaisesRegex(ValueError, "No individuals began the treatment strategy .*"):
            estimation_model.preprocess_data(self.df.assign(t=0))

    def test_preprocess_data_no_faults(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaisesRegex(ValueError, "No safe runs with outcome. Cannot perform estimation."):
            estimation_model.preprocess_data(self.df.assign(ok=False))

    def test_preprocess_data_only_faults(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaisesRegex(ValueError, "No faults with outcome. Cannot perform estimation."):
            estimation_model.preprocess_data(self.df.assign(ok=True))

    def test_preprocess_data_missing_control(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy + [(0, "invalid", 1)],
            treatment_strategy=self.treatment_strategy,
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaisesRegex(ValueError, "Missing data for control strategy."):
            estimation_model.preprocess_data(self.df.assign(ok=True))

    def test_preprocess_data_missing_treatment(self):
        estimation_model = IPCWEstimator(
            timesteps_per_observation=self.timesteps_per_observation,
            control_strategy=self.control_strategy,
            treatment_strategy=self.treatment_strategy + [(0, "invalid", 1)],
            outcome_variable="outcome",
            status_column=self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        with self.assertRaisesRegex(ValueError, "Missing data for treatment strategy."):
            estimation_model.preprocess_data(self.df.assign(ok=True))
