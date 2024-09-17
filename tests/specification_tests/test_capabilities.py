import unittest
from causal_testing.specification.capabilities import Capability, TreatmentSequence


class TestCapability(unittest.TestCase):
    """
    Test the Capability class for basic methods.
    """

    def setUp(self) -> None:
        pass

    def test_repr(self):
        cap = Capability("v", 1, 0, 1)
        self.assertEqual(str(cap), "(v, 1, 0-1)")


class TestTreatmentSequence(unittest.TestCase):
    """
    Test the TreatmentSequence class for basic methods.
    """

    def setUp(self) -> None:
        self.timesteps_per_intervention = 1

    def test_set_value(self):
        treatment_strategy = TreatmentSequence(self.timesteps_per_intervention, [("t", 1), ("t", 1), ("t", 1)])
        treatment_strategy.set_value(0, 0)
        self.assertEqual([x.value for x in treatment_strategy.capabilities], [0, 1, 1])

    def test_copy(self):
        control_strategy = TreatmentSequence(self.timesteps_per_intervention, [("t", 1), ("t", 1), ("t", 1)])
        treatment_strategy = control_strategy.copy()
        treatment_strategy.set_value(0, 0)
        self.assertEqual([x.value for x in control_strategy.capabilities], [1, 1, 1])
        self.assertEqual([x.value for x in treatment_strategy.capabilities], [0, 1, 1])
