import unittest
from causal_testing.testing.causal_test_outcome import CausalTestResult, ExactValue

class TestCausalTestOutcome(unittest.TestCase):

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
