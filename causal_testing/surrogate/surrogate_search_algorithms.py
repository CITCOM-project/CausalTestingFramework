"""Module containing implementation of search algorithm for surrogate search """
# pylint: disable=cell-var-from-loop
# Fitness functions are required to be iteratively defined, including all variables within.

from operator import itemgetter
from pygad import GA

from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import CubicSplineRegressionEstimator
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm, SearchFitnessFunction


class GeneticSearchAlgorithm(SearchAlgorithm):
    """Implementation of SearchAlgorithm class. Implements genetic search algorithm for surrogate models."""

    def __init__(self, delta=0.05, config: dict = None) -> None:
        super().__init__()

        self.delta = delta
        self.config = config
        self.contradiction_functions = {
            "positive": lambda x: -1 * x,
            "negative": lambda x: x,
            "no_effect": abs,
            "some_effect": lambda x: abs(1 / x),
        }

    def generate_fitness_functions(
        self, surrogate_models: list[CubicSplineRegressionEstimator]
    ) -> list[SearchFitnessFunction]:
        fitness_functions = []

        for surrogate in surrogate_models:
            contradiction_function = self.contradiction_functions[surrogate.expected_relationship]

            # The returned fitness function after including required variables into the function's scope
            def fitness_function(ga, solution, idx): # pylint: disable=unused-argument
                surrogate.control_value = solution[0] - self.delta
                surrogate.treatment_value = solution[0] + self.delta

                adjustment_dict = {}
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
            gene_types, gene_space = self.create_gene_types(fitness_function, specification)

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
                        raise ValueError(
                            "Gene space should not be set through config. This is generated from the causal "
                            "specification"
                        )
                    setattr(ga, k, v)

            ga.run()
            solution, fitness, _ = ga.best_solution()

            solution_dict = {}
            solution_dict[fitness_function.surrogate_model.treatment] = solution[0]
            for idx, adj in enumerate(fitness_function.surrogate_model.adjustment_set):
                solution_dict[adj] = solution[idx + 1]
            solutions.append((solution_dict, fitness, fitness_function.surrogate_model))

        return max(solutions, key=itemgetter(1))  # This can be done better with fitness normalisation between edges

    @staticmethod
    def create_gene_types(
        fitness_function: SearchFitnessFunction, specification: CausalSpecification
    ) -> tuple[list, list]:
        """Generate the gene_types and gene_space for a given fitness function and specification
        :param fitness_function: Instance of SearchFitnessFunction
        :param specification: The Causal Specification (combination of Scenario and Causal Dag)"""

        var_space = {}
        var_space[fitness_function.surrogate_model.treatment] = {}
        for adj in fitness_function.surrogate_model.adjustment_set:
            var_space[adj] = {}

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
        return gene_types, gene_space
