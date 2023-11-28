from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_suite import CausalTestSuite
from causal_testing.testing.estimators import Estimator

from dataclasses import dataclass


class SearchAlgorithm:
    def generate_fitness_functions(self, test_suite: CausalTestSuite, surrogate_models: list[Estimator]):
        pass

    def search(self) -> list:
        pass


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
        test_suite: CausalTestSuite,
        search_alogrithm: SearchAlgorithm,
        simulator: Simulator,
    ):
        self.test_suite = test_suite
        self.search_algorithm = search_alogrithm
        self.simulator = simulator

    def execute(self, data_collector: ObservationalDataCollector, max_executions: int = 200):
        data_collector.collect_data()

        for _i in range(max_executions):

            surrogate_models = []
            self.search_algorithm.generate_fitness_functions(self.test_suite, surrogate_models)
            candidate_test_case = self.search_algorithm.search()

            self.simulator.startup()
            test_result = self.simulator.run_with_config(candidate_test_case)
            self.simulator.shutdown()

            if test_result.fault:
                return test_result
            else:
                data_collector.data = data_collector.data.append(test_result.data, ignore_index=True)

        print("No fault found")
