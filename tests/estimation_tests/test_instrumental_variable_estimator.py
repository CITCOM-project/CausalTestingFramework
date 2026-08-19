import unittest

import numpy as np
import pandas as pd

from causal_testing.estimation.instrumental_variable_estimator import InstrumentalVariableEstimator


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
            treatment_variable="X",
            outcome_variable="Y",
            treatment_value=None,
            control_value=None,
            instrument="Z",
        )
        effect_estimate = iv_estimator.estimate_coefficient(self.df)
        self.assertEqual(effect_estimate.value[0], 2)
        self.assertEqual(effect_estimate.ci_low[0], 2)
        self.assertEqual(effect_estimate.ci_high[0], 2)

    def test_to_dict(self):
        iv_estimator = InstrumentalVariableEstimator(
            treatment_variable="X",
            outcome_variable="Y",
            control_value=0,
            treatment_value=1,
            instrument="Z",
        )
        self.assertEqual(
            iv_estimator.to_dict(),
            {
                "treatment_variable": "X",
                "outcome_variable": "Y",
                "alpha": 0.05,
                "control_value": 0,
                "treatment_value": 1,
                "instrument": "Z",
                "bootstrap_size": 100,
            },
        )
