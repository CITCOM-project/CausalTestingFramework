"""
This module implements a hill climbing algorithm to optimise causal DAGs based on the tests that pass/fail.
"""

import random
import time

import numpy as np
import pandas as pd

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.discovery.abstract_discovery import Discovery, TestResult


class HillClimberDiscovery(Discovery):
    """
    Simple hill climber evolution of cauasl DAGs via 1+1EA.
    Attempts to maximise the number of passing tests and minimise the number of failing tests.
    """

    def __init__(  # pylint: disable=R0917
        self,
        df: pd.DataFrame,
        random_seed: int = 0,
        included_edges: str = None,
        excluded_edges: str = None,
        max_iterations: int = 100,
        max_iterations_without_improvement: int = 10,
    ):
        super().__init__(df=df, random_seed=random_seed, included_edges=included_edges, excluded_edges=excluded_edges)
        self.max_iterations = int(max_iterations)
        self.max_iterations_without_improvement = int(max_iterations_without_improvement)

    def sum_test_outcomes(self, test_results: pd.DataFrame) -> dict:
        """
        Aggregate the number of passing, failing, and inestimable tests
        :param test_results: Dataframe containing the raw pass/fail/inestimable outcome of each test case.
        :returns: Dictionary containing the number of pass/fail/inestimable outcomes.
        """
        counts = pd.concat(
            [
                pd.DataFrame(np.sort(test_results[["treatment", "outcome"]], axis=1), columns=["treatment", "outcome"]),
                pd.get_dummies(test_results["result"]).astype(int),
            ],
            axis=1,
        )
        # Ensure every column is initialised - Test outcomes that never occurred won't be in the dataframe otherwise
        for col in TestResult:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts.groupby(["treatment", "outcome"]).sum().reset_index()[list(TestResult)]
        # The below line normalises by the number of tests *for each edge*
        # Independence tests X _||_ Y get two tests (X _||_ Y and Y _||_ X) because we don't know which way the
        # causality flows. We need to normalise this (e.g. if X _||_ Y and Y _||_ X both pass, then the score should be
        # 1 rather than 2) otherwise we end up unintentionally optimising for more independences.
        counts = counts.apply(lambda col: col / counts.sum(axis=1))

        return counts.sum(axis=0).to_dict()

    def evaluate_fitness(
        self,
        individual: CausalDAG,
    ) -> tuple[tuple[float, float, float], list[tuple[str, str]]]:
        """
        Evaluate the fitness of a given causal DAG by evaluating the corresponding test cases using a tier based
        fitness metric.
        lexicographical order (max pass, minimise failure, minimise unknown)
         e.g. (X pass, Y fail, Z+1 unknown) is better than (X pass, Y+1 fail, Z unknown)


        :param individual: The candidate individual to evaluate.
        :returns: Tuple of the form (X, Y), where X is a triple containing the number of passing, failing, and
                  inestimable tests respectively, and Y is a list of failing edges.
        """
        self.evaluate_tests(individual)
        counts = self.sum_test_outcomes(individual.test_results)

        # Add extra "var1" and "var2" columns to serve as order independent "treatment" and "outcome"
        query_df = pd.concat(
            [
                individual.test_results,
                pd.DataFrame(
                    np.sort(individual.test_results[["treatment", "outcome"]], axis=1), columns=["var1", "var2"]
                ),
            ],
            axis=1,
        )
        problem_tests = query_df.groupby(["var1", "var2"]).filter(
            # Groups are problematic if at least one test fails or no test passes
            lambda group: (group["result"] == TestResult.FAIL).any()
            or ~(group["result"] == TestResult.PASS).any()
        )
        problem_edges = problem_tests[["treatment", "outcome"]].apply(tuple, axis=1).tolist()

        fitness_values = (
            counts.get(TestResult.PASS, 0),
            -counts.get(TestResult.FAIL, 0),
            -counts.get(TestResult.INESTIMABLE, 0),
        )
        return fitness_values, problem_edges

    def discover(self) -> CausalDAG:
        """
        Discover the causal DAG.

        :returns: The inferred causal DAG.
        """

        start_time = time.time()
        individual = CausalDAG()
        individual.add_nodes_from(self.df.columns)
        individual.add_edges_from(self.possible_edges)
        self.remove_cycles(individual)
        fitness_values, problem_edges = self.evaluate_fitness(individual)

        iterations = self.max_iterations
        iterations_without_improvement = 0

        while problem_edges and iterations:
            iterations -= 1

            new_individual = individual.copy()
            for origin, dest in random.sample(
                # If we've gone over the maximum iterations without improvement
                problem_edges
                + (
                    self.possible_edges
                    if iterations_without_improvement > self.max_iterations_without_improvement
                    else []
                ),
                random.randint(1, len(problem_edges)),
            ):
                if new_individual.has_edge(origin, dest) and (origin, dest) not in self.included_edges:
                    new_individual.remove_edge(origin, dest)
                elif not new_individual.has_edge(origin, dest) and (origin, dest) not in self.excluded_edges:
                    # Want to bypass the cycle check of CausalDAG as we remove the cycles afterwards
                    new_individual.add_edge(origin, dest, ignore_cycles=True)
            self.remove_cycles(new_individual)
            new_fitness_values, new_problem_edges = self.evaluate_fitness(new_individual)

            if new_fitness_values > fitness_values:
                fitness_values = new_fitness_values
                problem_edges = new_problem_edges
                individual = new_individual
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

        end_time = time.time()
        individual.graph["fitness"] = fitness_values
        individual.graph["time"] = round(end_time - start_time)

        return individual
