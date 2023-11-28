from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.estimators import Estimator, PolynomialRegressionEstimator

from dataclasses import dataclass
from typing import Callable


class SearchAlgorithm:
    def generate_fitness_functions(self, specification: CausalSpecification, surrogate_models: list[Estimator]):
        pass

    def search(self, fitness_functions) -> list:
        pass

class GeneticSearchAlgorithm(SearchAlgorithm):
    def __init__(self, delta = 0.05) -> None:
        super().__init__()

        self.delta = delta

    def generate_fitness_functions(
        self, surrogate_models: list[PolynomialRegressionEstimator]
    ) -> Callable[[list[float], int], float]:
        fitness_functions = []

        for surrogate in surrogate_models:

            def fitness_function(solution, idx):
                surrogate.control_value = solution[0] - self.delta
                surrogate.treatment_value = solution[0] + self.delta

                adjustment_dict = dict()
                for i, adjustment in enumerate(surrogate.adjustment_set):
                    adjustment_dict[adjustment] = solution[i + 1]

                ate = surrogate.estimate_ate_calculated(adjustment_dict)[0]

                return ate  # TODO Need a way here of assessing if high or low ATE is correct

            fitness_functions.append(fitness_function)

        return fitness_functions
    
    def search(self, fitness_functions) -> list:
        pass # TODO Implement GA search


@dataclass
class SimulationResult:
    data: dict
    fault: bool


class Simulator:
    def startup(self, **kwargs):
        pass

    def shutdown(self, **kwargs):
        pass

    def run_with_config(self, configuration) -> SimulationResult:
        pass


class CausalSurrogateAssistedTestCase:
    def __init__(
        self,
        specification: CausalSpecification,
        search_alogrithm: SearchAlgorithm,
        simulator: Simulator,
    ):
        self.specification = specification
        self.search_algorithm = search_alogrithm
        self.simulator = simulator

    def execute(
        self,
        data_collector: ObservationalDataCollector,
        max_executions: int = 200,
        custom_data_aggregator: Callable[[dict, dict], dict] = None,
    ):
        data_collector.collect_data()

        for _i in range(max_executions):
            surrogate_models = self.generate_surrogates(self.specification, data_collector)
            fitness_functions = self.search_algorithm.generate_fitness_functions(surrogate_models)
            candidate_test_case = self.search_algorithm.search(fitness_functions)

            self.simulator.startup()
            test_result = self.simulator.run_with_config(candidate_test_case)
            self.simulator.shutdown()

            if test_result.fault:
                return test_result
            else:
                if custom_data_aggregator is not None:
                    data_collector.data = custom_data_aggregator(data_collector.data, test_result.data)
                else:
                    data_collector.data = data_collector.data.append(test_result.data, ignore_index=True)

        print("No fault found")

    def generate_surrogates(
        self, specification: CausalSpecification, data_collector: ObservationalDataCollector
    ) -> list[PolynomialRegressionEstimator]:
        surrogate_models = []

        for u, v in specification.causal_dag.edges:
            base_test_case = BaseTestCase(u, v)
            minimal_adjustment_set = specification.causal_dag.identification(base_test_case)

            surrogate = PolynomialRegressionEstimator(u, 0, 0, minimal_adjustment_set, v, 4, df=data_collector.data)
            surrogate_models.append(surrogate)

        return surrogate_models
