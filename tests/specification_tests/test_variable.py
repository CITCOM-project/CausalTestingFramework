import unittest
from enum import Enum
from scipy.stats import norm, kstest

from causal_testing.specification.variable import Variable, Input


class TestVariable(unittest.TestCase):
    """
    Test the Variable class for basic methods.
    """

    def setUp(self) -> None:
        pass

    def test_sample_flakey(self):
        ip = Input("ip", float, norm)
        self.assertGreater(kstest(ip.sample(10), norm.cdf).pvalue, 0.95)

    def test_typestring(self):
        class Var(Variable):
            pass

        var = Var("v", int)
        self.assertEqual(var.typestring(), "Var")

    def test_copy(self):
        ip = Input("ip", float, norm)
        self.assertTrue(ip.copy() is not ip)
        self.assertEqual(ip.copy().name, ip.name)
        self.assertEqual(ip.copy().datatype, ip.datatype)
        self.assertEqual(ip.copy().distribution, ip.distribution)
        self.assertEqual(repr(ip), repr(ip.copy()))
