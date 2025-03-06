import unittest
import pandas as pd
from causal_testing.specification.variable import Input, Output

from causal_testing.estimation.ipcw_estimator import IPCWEstimator


class TestIPCWEstimator(unittest.TestCase):
    """
    Test the IPCW estimator class
    """

    def setUp(self) -> None:
        self.outcome = Output("outcome", float)
        self.status_column = "ok"
        self.timesteps_per_intervention = 1
        self.control_strategy = [[t, "t", 0] for t in range(1, 4, self.timesteps_per_intervention)]
        self.treatment_strategy = [[t, "t", 1] for t in range(1, 4, self.timesteps_per_intervention)]
        self.fit_bl_switch_formula = "xo_t_do ~ time"
        self.df = pd.read_csv("tests/resources/data/temporal_data.csv")
        self.df[self.status_column] = self.df["outcome"] == 1

    def test_estimate_hazard_ratio(self):
        estimation_model = IPCWEstimator(
            self.df,
            self.timesteps_per_intervention,
            self.control_strategy,
            self.treatment_strategy,
            self.outcome,
            self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        estimate, _ = estimation_model.estimate_hazard_ratio()
        self.assertEqual(round(estimate["trtrand"], 3), 1.351)

    def test_invalid_treatment_strategies(self):
        with self.assertRaises(ValueError):
            IPCWEstimator(
                self.df.assign(t=(["1", "0"] * len(self.df))[: len(self.df)]),
                self.timesteps_per_intervention,
                self.control_strategy,
                self.treatment_strategy,
                self.outcome,
                self.status_column,
                fit_bl_switch_formula=self.fit_bl_switch_formula,
                fit_bltd_switch_formula=self.fit_bl_switch_formula,
                eligibility=None,
            )

    def test_invalid_fault_t_do(self):
        estimation_model = IPCWEstimator(
            self.df.assign(outcome=1),
            self.timesteps_per_intervention,
            self.control_strategy,
            self.treatment_strategy,
            self.outcome,
            self.status_column,
            fit_bl_switch_formula=self.fit_bl_switch_formula,
            fit_bltd_switch_formula=self.fit_bl_switch_formula,
            eligibility=None,
        )
        estimation_model.df["fault_t_do"] = 0
        with self.assertRaises(ValueError):
            estimation_model.estimate_hazard_ratio()

    def test_no_individual_began_control_strategy(self):
        with self.assertRaises(ValueError):
            IPCWEstimator(
                self.df.assign(t=1),
                self.timesteps_per_intervention,
                self.control_strategy,
                self.treatment_strategy,
                self.outcome,
                self.status_column,
                fit_bl_switch_formula=self.fit_bl_switch_formula,
                fit_bltd_switch_formula=self.fit_bl_switch_formula,
                eligibility=None,
            )

    def test_no_individual_began_treatment_strategy(self):
        with self.assertRaises(ValueError):
            IPCWEstimator(
                self.df.assign(t=0),
                self.timesteps_per_intervention,
                self.control_strategy,
                self.treatment_strategy,
                self.outcome,
                self.status_column,
                fit_bl_switch_formula=self.fit_bl_switch_formula,
                fit_bltd_switch_formula=self.fit_bl_switch_formula,
                eligibility=None,
            )
