from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.estimators import Estimator, PolynomialRegressionEstimator

from pygad import GA

from dataclasses import dataclass
from typing import Callable
from operator import itemgetter


@dataclass
class SimulationResult:
    data: dict
    fault: bool


@dataclass
class SearchFitnessFunction:
    fitness_function: Callable
    surrogate_model: PolynomialRegressionEstimator


class SearchAlgorithm:
    def generate_fitness_functions(self, surrogate_models: list[Estimator]) -> list[SearchFitnessFunction]:
        pass

    def search(self, fitness_functions: list[SearchFitnessFunction], specification: CausalSpecification) -> list:
        pass


class GeneticSearchAlgorithm(SearchAlgorithm):
    def __init__(self, delta=0.05, config: dict = None) -> None:
        super().__init__()

        self.delta = delta
        self.config = config
        self.contradiction_functions = {
            "positive": lambda x: -1 * x,
            "negative": lambda x: x,
            "no effect": lambda x: abs(x),
            "some effect": lambda x: abs(1 / x),
        }

    def generate_fitness_functions(
        self, surrogate_models: list[PolynomialRegressionEstimator]
    ) -> list[SearchFitnessFunction]:
        fitness_functions = []

        for surrogate, expected in surrogate_models:
            contradiction_function = self.contradiction_functions[expected]

            def fitness_function(_ga, solution, idx):
                surrogate.control_value = solution[0] - self.delta
                surrogate.treatment_value = solution[0] + self.delta

                adjustment_dict = dict()
                for i, adjustment in enumerate(surrogate.adjustment_set):
                    adjustment_dict[adjustment] = solution[i + 1]

                ate = surrogate.estimate_ate_calculated(adjustment_dict)

                return contradiction_function(ate)

            search_fitness_function = SearchFitnessFunction(fitness_function, surrogate)

            fitness_functions.append(search_fitness_function)

        return fitness_functions

    def search(self, fitness_functions: list[SearchFitnessFunction], specification: CausalSpecification) -> list:
        solutions = []

        for fitness_function in fitness_functions:
            var_space = dict()
            var_space[fitness_function.surrogate_model.treatment] = dict()
            for adj in fitness_function.surrogate_model.adjustment_set:
                var_space[adj] = dict()

            for relationship in list(specification.scenario.constraints):
                rel_split = str(relationship).split(" ")

                if rel_split[1] == ">=":
                    var_space[rel_split[0]]["low"] = int(rel_split[2])
                elif rel_split[1] == "<=":
                    var_space[rel_split[0]]["high"] = int(rel_split[2])

            gene_space = []
            gene_space.append(var_space[fitness_function.surrogate_model.treatment])
            for adj in fitness_function.surrogate_model.adjustment_set:
                gene_space.append(var_space[adj])

            ga = GA(
                num_generations=200,
                num_parents_mating=4,
                fitness_func=fitness_function.fitness_function,
                sol_per_pop=10,
                num_genes=1 + len(fitness_function.surrogate_model.adjustment_set),
                gene_space=gene_space,
            )

            if self.config is not None:
                for k, v in self.config.items():
                    if k == "gene_space":
                        raise Exception(
                            "Gene space should not be set through config. This is generated from the causal specification"
                        )
                    setattr(ga, k, v)

            ga.run()
            solution, fitness, _idx = ga.best_solution()

            solution_dict = dict()
            solution_dict[fitness_function.surrogate_model.treatment] = solution[0]
            for idx, adj in enumerate(fitness_function.surrogate_model.adjustment_set):
                solution_dict[adj] = solution[idx + 1]
            solutions.append((solution_dict, fitness))

        return max(solutions, key=itemgetter(1))[0]


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
            candidate_test_case = self.search_algorithm.search(fitness_functions, self.specification)

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
    ) -> list[SearchFitnessFunction]:
        surrogate_models = []

        for u, v in specification.causal_dag.graph.edges:
            edge_metadata = specification.causal_dag.graph.adj[u][v]
            if "included" in edge_metadata:
                from_var = specification.scenario.variables.get(u)
                to_var = specification.scenario.variables.get(v)
                base_test_case = BaseTestCase(from_var, to_var)

                minimal_adjustment_set = specification.causal_dag.identification(base_test_case, specification.scenario)

                surrogate = PolynomialRegressionEstimator(u, 0, 0, minimal_adjustment_set, v, 4, df=data_collector.data)
                surrogate_models.append((surrogate, edge_metadata["expected"]))

        return surrogate_models
