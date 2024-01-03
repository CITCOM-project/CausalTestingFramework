import unittest
from causal_testing.surrogate.causal_surrogate_assisted import SimulationResult, SearchFitnessFunction
from causal_testing.testing.estimators import Estimator, CubicSplineRegressionEstimator

class TestSimulationResult(unittest.TestCase):

    def setUp(self):

        self.data = {'key': 'value'}

    def test_inputs(self):

        fault_values = [True, False]

        relationship_values = ["positive", "negative", None]

        for fault in fault_values:

            for relationship in relationship_values:
                with self.subTest(fault=fault, relationship=relationship):
                    result = SimulationResult(data=self.data, fault=fault, relationship=relationship)

                    self.assertIsInstance(result.data, dict)

                    self.assertEqual(result.fault, fault)

                    self.assertEqual(result.relationship, relationship)

class TestSearchFitnessFunction(unittest.TestCase):

    #TODO: complete tests for causal surrogate

    def test_init_valid_values(self):

        test_function = lambda x: x **2

        surrogate_model = CubicSplineRegressionEstimator()

        search_function = SearchFitnessFunction(fitness_function=test_function, surrogate_model=surrogate_model)

        self.assertIsCallable(search_function.fitness_function)
        self.assertIsInstance(search_function.surrogate_model, CubicSplineRegressionEstimator)