"""
This module implements a multiobjective discovery algorithm in terms of test outcomes.
"""

import numpy as np
import pandas as pd
import pygad

from causal_testing.discovery.abstract_discovery import Discovery
from causal_testing.specification.causal_dag import CausalDAG


class NSGADiscovery(Discovery):
    """
    Multiobjective evolution of cauasl DAGs via NSGA2.
    Attempts to optimise the number of passing tests, where each possible relationship is a "feature".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        random_seed: int = 0,
        include_edges: str = None,
        exclude_edges: str = None,
        alpha: float = 0.05,
        max_iterations: int = 100,
        num_parents_mating: int = 2,
        population_size: int = 5,  # Population size
    ):
        super().__init__(
            df=df, random_seed=random_seed, include_edges=include_edges, exclude_edges=exclude_edges, alpha=alpha
        )
        self.max_iterations = int(max_iterations)
        self.num_parents_mating = int(num_parents_mating)
        self.sol_per_pop = int(population_size)

    def binary_string_to_causal_dag(self, individual: np.array) -> CausalDAG:
        """
        Converts a binary string representation of a causal DAG to a CausalDAG object.

        :param individual: Bitstring of the same length as `possible_edges` such that 1 at position `i` represents
        possible_edges[i] being an edge in the graph and 0 represents it not being.
        :returns: The converted CausalDAG instance.
        """
        causal_dag = CausalDAG()
        origins, destinations = zip(*self.possible_edges)
        causal_dag.add_nodes_from(set(origins).union(set(destinations)))
        causal_dag.add_edges_from([edge for edge, add in zip(self.possible_edges, individual) if add])
        return causal_dag

    def causal_dag_to_binary_string(self, causal_dag: CausalDAG) -> np.array:
        """
        Converts a CausalDAG to a binary string representation.

        :param causal_dag: The CausalDAG to convert.
        :returns: The converted binary string such that 1 at position `i` represents possible_edges[i] being an edge in
                  the graph and 0 represents it not being.
        """
        return np.array([int(edge in causal_dag.edges) for edge in self.possible_edges])

    def multi_objective_fitness(
        self,
        ga_instance: pygad.GA,  # pylint: disable=W0613
        individual: np.array,
        individual_inx: int,  # pylint: disable=W0613
    ) -> np.array:
        """
        Remove cycles and calculate the multi-objective fitness of the resulting causal DAG in terms of tests passing
        failing, and being inestimable.
        NOTE: this is in terms of the number of possible (X, Y) *relationships* rather than edges, so is not
        order dependent. I.e. (X, Y) and (Y, X) are the same. This stops the algorithm optimising for independences,
        which get two tests (one in each direction).

        :param ga_instance: The calling GA instance. NOT USED - required for compatibility.
        :param individual: The individual to evaluate.
        :param individual_inx: The index of the individual in the population. NOT USED - required for compatibility.
        :returns: Numeric numpy array representing the outcome of each test.
        """
        # Repair by removing cycles
        causal_dag = self.binary_string_to_causal_dag(individual)
        self.remove_cycles(causal_dag)
        individual[:] = self.causal_dag_to_binary_string(causal_dag)

        test_results = self.evaluate_tests(causal_dag)
        test_results["result"] = test_results["result"].apply(lambda x: x.value).astype(float)

        # Sort so that the "treatment" is always the first item alphabetically
        # This ensures that the fitness vector always represents the same features
        test_results[["treatment", "outcome"]] = np.sort(test_results[["treatment", "outcome"]], axis=1)
        # We then need to normalise each potential edge since independences get two tests (X _||_ Y and Y _||_ X)
        # because we don't know which way the causality flows.
        # "groupby" will sort by (treatment, outcome)
        result = test_results.groupby(["treatment", "outcome"], sort=True)["result"].mean()

        return result.values

    def discover(self) -> CausalDAG:
        """
        Discover the causal DAG.

        :returns: The inferred causal DAG.
        """

        gene_space = []
        for edge in self.possible_edges:
            # Excluded edges are not in possible_edges, so no need to explicitly test for this
            if edge in self.include_edges:
                # Must be included
                gene_space.append([1])
            else:
                # Free to evolve
                gene_space.append([0, 1])

        ga_instance = pygad.GA(
            num_generations=self.max_iterations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.sol_per_pop,  # Population size
            num_genes=len(self.possible_edges),
            gene_space=gene_space,
            gene_type=int,
            fitness_func=self.multi_objective_fitness,
            parent_selection_type="nsga2",
            random_seed=self.random_seed,
        )
        ga_instance.run()

        best_solution, _, _ = ga_instance.best_solution()
        best_individual = self.binary_string_to_causal_dag(best_solution)
        self.evaluate_tests(best_individual)

        return best_individual
