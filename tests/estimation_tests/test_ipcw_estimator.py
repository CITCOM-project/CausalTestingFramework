import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.variable import Input
from causal_testing.utils.validation import CausalValidator
from causal_testing.specification.capabilities import TreatmentSequence

from causal_testing.estimation.ipcw_estimator import IPCWEstimator


class TestIPCWEstimator(unittest.TestCase):
    """
    Test the IPCW estimator class
    """

    def test_estimate_hazard_ratio(self):
        timesteps_per_intervention = 1
        control_strategy = TreatmentSequence(timesteps_per_intervention, [("t", 0), ("t", 0), ("t", 0)])
        treatment_strategy = TreatmentSequence(timesteps_per_intervention, [("t", 1), ("t", 1), ("t", 1)])
        outcome = "outcome"
        fit_bl_switch_formula = "xo_t_do ~ time"
        df = pd.read_csv("tests/resources/data/temporal_data.csv")
        df["ok"] = df["outcome"] == 1
        estimation_model = IPCWEstimator(
            df,
            timesteps_per_intervention,
            control_strategy,
            treatment_strategy,
            outcome,
            "ok",
            fit_bl_switch_formula=fit_bl_switch_formula,
            fit_bltd_switch_formula=fit_bl_switch_formula,
            eligibility=None,
        )
        estimate, intervals = estimation_model.estimate_hazard_ratio()
        self.assertEqual(estimate["trtrand"], 1.0)

    def test_invalid_treatment_strategies(self):
        timesteps_per_intervention = 1
        control_strategy = TreatmentSequence(timesteps_per_intervention, [("t", 0), ("t", 0), ("t", 0)])
        treatment_strategy = TreatmentSequence(timesteps_per_intervention, [("t", 1), ("t", 1), ("t", 1)])
        outcome = "outcome"
        fit_bl_switch_formula = "xo_t_do ~ time"
        df = pd.read_csv("tests/resources/data/temporal_data.csv")
        df["t"] = (["1", "0"] * len(df))[: len(df)]
        df["ok"] = df["outcome"] == 1
        with self.assertRaises(ValueError):
            estimation_model = IPCWEstimator(
                df,
                timesteps_per_intervention,
                control_strategy,
                treatment_strategy,
                outcome,
                "ok",
                fit_bl_switch_formula=fit_bl_switch_formula,
                fit_bltd_switch_formula=fit_bl_switch_formula,
                eligibility=None,
            )

    def test_invalid_fault_t_do(self):
        timesteps_per_intervention = 1
        control_strategy = TreatmentSequence(timesteps_per_intervention, [("t", 0), ("t", 0), ("t", 0)])
        treatment_strategy = TreatmentSequence(timesteps_per_intervention, [("t", 1), ("t", 1), ("t", 1)])
        outcome = "outcome"
        fit_bl_switch_formula = "xo_t_do ~ time"
        df = pd.read_csv("tests/resources/data/temporal_data.csv")
        df["ok"] = df["outcome"] == 1
        estimation_model = IPCWEstimator(
            df,
            timesteps_per_intervention,
            control_strategy,
            treatment_strategy,
            outcome,
            "ok",
            fit_bl_switch_formula=fit_bl_switch_formula,
            fit_bltd_switch_formula=fit_bl_switch_formula,
            eligibility=None,
        )
        estimation_model.df["fault_t_do"] = 0
        with self.assertRaises(ValueError):
            estimate, intervals = estimation_model.estimate_hazard_ratio()
