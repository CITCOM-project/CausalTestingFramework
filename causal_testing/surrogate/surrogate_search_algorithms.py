from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import Estimator, PolynomialRegressionEstimator
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm, SearchFitnessFunction

from pygad import GA
from operator import itemgetter


class GeneticSearchAlgorithm(SearchAlgorithm):
    def __init__(self, delta=0.05, config: dict = None) -> None:
        super().__init__()

        self.delta = delta
        self.config = config
        self.contradiction_functions = {
            "positive": lambda x: -1 * x,
            "negative": lambda x: x,
            "no_effect": lambda x: abs(x),
            "some_effect": lambda x: abs(1 / x),
        }

    def generate_fitness_functions(
        self, surrogate_models: list[PolynomialRegressionEstimator]
    ) -> list[SearchFitnessFunction]:
        fitness_functions = []

        for surrogate in surrogate_models:
            contradiction_function = self.contradiction_functions[surrogate.expected_relationship]

            # The returned fitness function after including required variables into the function's scope
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

            gene_types = []
            gene_types.append(specification.scenario.variables.get(fitness_function.surrogate_model.treatment).datatype)
            for adj in fitness_function.surrogate_model.adjustment_set:
                gene_types.append(specification.scenario.variables.get(adj).datatype)

            ga = GA(
                num_generations=200,
                num_parents_mating=4,
                fitness_func=fitness_function.fitness_function,
                sol_per_pop=10,
                num_genes=1 + len(fitness_function.surrogate_model.adjustment_set),
                gene_space=gene_space,
                gene_type=gene_types,
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
            solutions.append((solution_dict, fitness, fitness_function.surrogate_model))

        return max(
            solutions, key=itemgetter(1)
        )  # TODO This can be done better with fitness normalisation between edges
