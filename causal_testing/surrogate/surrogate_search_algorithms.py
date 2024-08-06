"""Module containing implementation of search algorithm for surrogate search """

# Fitness functions are required to be iteratively defined, including all variables within.

from operator import itemgetter
from pygad import GA

from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.estimation.cubic_spline_estimator import CubicSplineRegressionEstimator
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm


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

    # pylint: disable=too-many-locals
    def search(
        self, surrogate_models: list[CubicSplineRegressionEstimator], specification: CausalSpecification
    ) -> list:
        solutions = []

        for surrogate in surrogate_models:
            contradiction_function = self.contradiction_functions[surrogate.expected_relationship]

            # The GA fitness function after including required variables into the function's scope
            # Unused arguments are required for pygad's fitness function signature
            # pylint: disable=cell-var-from-loop
            def fitness_function(ga, solution, idx):  # pylint: disable=unused-argument
                surrogate.control_value = solution[0] - self.delta
                surrogate.treatment_value = solution[0] + self.delta

                adjustment_dict = {}
                for i, adjustment in enumerate(surrogate.adjustment_set):
                    adjustment_dict[adjustment] = solution[i + 1]

                ate = surrogate.estimate_ate_calculated(adjustment_dict)
                if len(ate) > 1:
                    raise ValueError(
                        "Multiple ate values provided but currently only single values supported in this method"
                    )
                return contradiction_function(ate[0])

            gene_types, gene_space = self.create_gene_types(surrogate, specification)

            ga = GA(
                num_generations=200,
                num_parents_mating=4,
                fitness_func=fitness_function,
                sol_per_pop=10,
                num_genes=1 + len(surrogate.adjustment_set),
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
            solution_dict[surrogate.treatment] = solution[0]
            for idx, adj in enumerate(surrogate.adjustment_set):
                solution_dict[adj] = solution[idx + 1]
            solutions.append((solution_dict, fitness, surrogate))

        return max(solutions, key=itemgetter(1))  # This can be done better with fitness normalisation between edges

    @staticmethod
    def create_gene_types(
        surrogate_model: CubicSplineRegressionEstimator, specification: CausalSpecification
    ) -> tuple[list, list]:
        """Generate the gene_types and gene_space for a given fitness function and specification
        :param surrogate_model: Instance of a CubicSplineRegressionEstimator
        :param specification: The Causal Specification (combination of Scenario and Causal Dag)"""

        var_space = {}
        var_space[surrogate_model.treatment] = {}
        for adj in surrogate_model.adjustment_set:
            var_space[adj] = {}

        for relationship in list(specification.scenario.constraints):
            rel_split = str(relationship).split(" ")

            if rel_split[0] in var_space:
                datatype = specification.scenario.variables.get(rel_split[0]).datatype
                if rel_split[1] == ">=":
                    var_space[rel_split[0]]["low"] = datatype(rel_split[2])
                elif rel_split[1] == "<=":
                    if datatype == int:
                        var_space[rel_split[0]]["high"] = int(rel_split[2]) + 1
                    else:
                        var_space[rel_split[0]]["high"] = datatype(rel_split[2])

        gene_space = []
        gene_space.append(var_space[surrogate_model.treatment])
        for adj in surrogate_model.adjustment_set:
            gene_space.append(var_space[adj])

        gene_types = []
        gene_types.append(specification.scenario.variables.get(surrogate_model.treatment).datatype)
        for adj in surrogate_model.adjustment_set:
            gene_types.append(specification.scenario.variables.get(adj).datatype)
        return gene_types, gene_space
