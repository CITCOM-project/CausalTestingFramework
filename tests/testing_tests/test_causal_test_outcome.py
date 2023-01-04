import unittest
from causal_testing.testing.causal_test_outcome import ExactValue, SomeEffect
from causal_testing.testing.causal_test_result import CausalTestResult

class TestCausalTestOutcome(unittest.TestCase):
    """ Test the TestCausalTestOutcome basic methods.
    """

    def test_empty_adjustment_set(self):
        ctr = CausalTestResult(treatment="A", outcome="A", treatment_value=1,
                     control_value=0, adjustment_set={}, ate=0,
                     confidence_intervals = None, effect_modifier_configuration = None)

        self.assertIsNone(ctr.ci_low())
        self.assertIsNone(ctr.ci_high())


    def test_exactValue_pass(self):
        ctr = CausalTestResult(treatment="A", outcome="A", treatment_value=1,
                     control_value=0, adjustment_set={}, ate=5.05,
                     confidence_intervals = None, effect_modifier_configuration = None)
        ev = ExactValue(5, 0.1)
        self.assertTrue(ev.apply(ctr))


    def test_exactValue_fail(self):
        ctr = CausalTestResult(treatment="A", outcome="A", treatment_value=1,
                     control_value=0, adjustment_set={}, ate=0,
                     confidence_intervals = None, effect_modifier_configuration = None)
        ev = ExactValue(5, 0.1)
        self.assertFalse(ev.apply(ctr))


    def test_someEffect_pass(self):
        ctr = CausalTestResult(treatment="A", outcome="A", treatment_value=1,
                     control_value=0, adjustment_set={}, ate=5.05,
                     confidence_intervals = [4.8, 6.7], effect_modifier_configuration = None)
        ev = SomeEffect()
        self.assertTrue(ev.apply(ctr))


    def test_someEffect_fail(self):
        ctr = CausalTestResult(treatment="A", outcome="A", treatment_value=1,
                     control_value=0, adjustment_set={}, ate=0,
                     confidence_intervals = [-0.1, 0.2], effect_modifier_configuration = None)
        ev = SomeEffect()
        self.assertFalse(ev.apply(ctr))
