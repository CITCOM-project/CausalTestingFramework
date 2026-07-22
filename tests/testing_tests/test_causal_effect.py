import unittest
import pandas as pd
from causal_testing.testing.causal_effect import ExactValue, SomeEffect, Positive, Negative, NoEffect
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.utils.validation import CausalValidator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output


class TestCausalEffect(unittest.TestCase):
    """Test the TestCausalEffect basic methods."""

    def setUp(self) -> None:
        base_test_case = BaseTestCase(Input("A", float), Output("A", float))
        self.estimator = LinearRegressionEstimator(
            base_test_case=base_test_case,
            treatment_value=1,
            control_value=0,
            adjustment_set={},
        )

    def test_None_ci(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0)),
        )

        self.assertIsNone(ctr.effect_estimate.ci_low)
        self.assertIsNone(ctr.effect_estimate.ci_high)

    def test_empty_adjustment_set(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0)),
        )

        self.assertIsNone(ctr.effect_estimate.ci_low)
        self.assertIsNone(ctr.effect_estimate.ci_high)

    def test_Positive_ate_pass(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="ate", value=pd.Series(5.05), ci_low=pd.Series(5), ci_high=pd.Series(6)
            ),
        )
        ev = Positive()
        self.assertTrue(ev.apply(ctr))

    def test_Positive_risk_ratio_pass(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="risk_ratio", value=pd.Series(5.05), ci_low=pd.Series(5), ci_high=pd.Series(6)
            ),
        )
        ev = Positive()
        self.assertTrue(ev.apply(ctr))

    def test_Positive_fail(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0), ci_low=pd.Series(-1), ci_high=pd.Series(1)),
        )
        ev = Positive()
        self.assertFalse(ev.apply(ctr))

    def test_Negative_ate_pass(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="ate", value=pd.Series(-5.05), ci_low=pd.Series(-6), ci_high=pd.Series(-5)
            ),
        )
        ev = Negative()
        self.assertTrue(ev.apply(ctr))

    def test_Negative_risk_ratio_pass(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="risk_ratio", value=pd.Series(0.2), ci_low=pd.Series(0.1), ci_high=pd.Series(0.5)
            ),
        )
        ev = Negative()
        self.assertTrue(ev.apply(ctr))

    def test_Negative_fail(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0), ci_low=pd.Series(-1), ci_high=pd.Series(1)),
        )
        ev = Negative()
        self.assertFalse(ev.apply(ctr))

    def test_exactValue_pass(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(5.05)),
        )
        ev = ExactValue(5, 0.1)
        self.assertTrue(ev.apply(ctr))

    def test_exactValue_pass_ci(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="ate", value=pd.Series(5.05), ci_low=pd.Series(4), ci_high=pd.Series(6)
            ),
        )
        ev = ExactValue(5, 0.1)
        self.assertTrue(ev.apply(ctr))

    def test_exactValue_ci_pass_ci(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="ate", value=pd.Series(5.05), ci_low=pd.Series(4.1), ci_high=pd.Series(5.9)
            ),
        )
        ev = ExactValue(5, ci_low=4, ci_high=6)
        self.assertTrue(ev.apply(ctr))

    def test_exactValue_ci_fail_ci(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="ate", value=pd.Series(5.05), ci_low=pd.Series(3.9), ci_high=pd.Series(6.1)
            ),
        )
        ev = ExactValue(5, ci_low=4, ci_high=6)
        self.assertFalse(ev.apply(ctr))

    def test_exactValue_fail(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0)),
        )
        ev = ExactValue(5, 0.1)
        self.assertFalse(ev.apply(ctr))

    def test_invalid_atol(self):
        with self.assertRaises(ValueError):
            ExactValue(5, -0.1)

    def test_unspecified_ci_high(self):
        with self.assertRaises(ValueError):
            ExactValue(5, ci_low=-0.1)

    def test_unspecified_ci_low(self):
        with self.assertRaises(ValueError):
            ExactValue(5, ci_high=-0.1)

    def test_invalid_ci_range(self):
        with self.assertRaises(ValueError):
            ExactValue(5, ci_low=6, ci_high=7, atol=0.05)

    def test_invalid_ci_atol(self):
        with self.assertRaises(ValueError):
            ExactValue(1000, ci_low=999, ci_high=1001, atol=50)

    def test_invalid(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="invalid", value=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
            ),
        )
        with self.assertRaises(ValueError):
            SomeEffect().apply(ctr)
        with self.assertRaises(ValueError):
            NoEffect().apply(ctr)
        with self.assertRaises(ValueError):
            Positive().apply(ctr)
        with self.assertRaises(ValueError):
            Negative().apply(ctr)

    def test_someEffect_pass_coefficient(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="coefficient", value=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
            ),
        )
        self.assertTrue(SomeEffect().apply(ctr))
        self.assertFalse(NoEffect().apply(ctr))

    def test_someEffect_pass_ate(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="coefficient", value=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
            ),
        )
        self.assertTrue(SomeEffect().apply(ctr))
        self.assertFalse(NoEffect().apply(ctr))

    def test_someEffect_pass_rr(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="coefficient", value=pd.Series(5.05), ci_low=pd.Series(4.8), ci_high=pd.Series(6.7)
            ),
        )
        self.assertTrue(SomeEffect().apply(ctr))
        self.assertFalse(NoEffect().apply(ctr))

    def test_someEffect_fail(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(
                type="ate", value=pd.Series(0), ci_low=pd.Series(-0.1), ci_high=pd.Series(0.2)
            ),
        )
        self.assertFalse(SomeEffect().apply(ctr))
        self.assertTrue(NoEffect().apply(ctr))

    def test_someEffect_None(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series(0)),
        )
        self.assertEqual(SomeEffect().apply(ctr), None)

    def test_positive_risk_ratio_e_value(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value(1.5)
        self.assertEqual(round(e_value, 4), 2.366)

    def test_positive_risk_ratio_e_value_using_ci(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value_using_ci(1.5, [1.2, 1.8])
        self.assertEqual(round(e_value, 4), 1.6899)

    def test_negative_risk_ratio_e_value(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value(0.8)
        self.assertEqual(round(e_value, 4), 1.809)

    def test_negative_risk_ratio_e_value_using_ci(self):
        cv = CausalValidator()
        e_value = cv.estimate_e_value_using_ci(0.8, [0.2, 0.9])
        self.assertEqual(round(e_value, 4), 1.4625)

    def test_multiple_value_exception_caught(self):
        ctr = CausalTestResult(
            effect_estimate=EffectEstimate(type="ate", value=pd.Series([0, 1])),
        )
        with self.assertRaises(ValueError):
            Positive().apply(ctr)
        with self.assertRaises(ValueError):
            Negative().apply(ctr)
