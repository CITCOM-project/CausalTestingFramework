"""
This module contains the ShouldCause and ShouldNotCause metamorphic relations as
defined in our ICST paper [https://eprints.whiterose.ac.uk/195317/].
"""

from dataclasses import dataclass
from typing import Iterable
from itertools import combinations
import logging
import json
from multiprocessing import Pool

import networkx as nx

from causal_testing.specification.causal_specification import CausalDAG, Node
from causal_testing.testing.base_test_case import BaseTestCase

logger = logging.getLogger(__name__)


@dataclass(order=True)
class MetamorphicRelation:
    """Class representing a metamorphic relation."""

    base_test_case: BaseTestCase
    adjustment_vars: Iterable[Node]

    def __eq__(self, other):
        same_type = self.__class__ == other.__class__
        same_treatment = self.base_test_case.treatment_variable == other.base_test_case.treatment_variable
        same_outcome = self.base_test_case.outcome_variable == other.base_test_case.outcome_variable
        same_effect = self.base_test_case.effect == other.base_test_case.effect
        same_adjustment_set = set(self.adjustment_vars) == set(other.adjustment_vars)
        return same_type and same_treatment and same_outcome and same_effect and same_adjustment_set


class ShouldCause(MetamorphicRelation):
    """Class representing a should cause metamorphic relation."""

    def to_json_stub(self, skip=True) -> dict:
        """Convert to a JSON frontend stub string for user customisation"""
        return {
            "name": str(self),
            "estimator": "LinearRegressionEstimator",
            "estimate_type": "coefficient",
            "effect": "direct",
            "treatment_variable": self.base_test_case.treatment_variable,
            "expected_effect": {self.base_test_case.outcome_variable: "SomeEffect"},
            "formula": (
                f"{self.base_test_case.outcome_variable} ~ "
                f"{' + '.join([self.base_test_case.treatment_variable] + self.adjustment_vars)}"
            ),
            "skip": skip,
        }

    def __str__(self):
        formatted_str = f"{self.base_test_case.treatment_variable} --> {self.base_test_case.outcome_variable}"
        if self.adjustment_vars:
            formatted_str += f" | {self.adjustment_vars}"
        return formatted_str


class ShouldNotCause(MetamorphicRelation):
    """Class representing a should cause metamorphic relation."""

    def to_json_stub(self, skip=True) -> dict:
        """Convert to a JSON frontend stub string for user customisation"""
        return {
            "name": str(self),
            "estimator": "LinearRegressionEstimator",
            "estimate_type": "coefficient",
            "effect": "direct",
            "treatment_variable": self.base_test_case.treatment_variable,
            "expected_effect": {self.base_test_case.outcome_variable: "NoEffect"},
            "formula": (
                f"{self.base_test_case.outcome_variable} ~ "
                f"{' + '.join([self.base_test_case.treatment_variable] + self.adjustment_vars)}"
            ),
            "alpha": 0.05,
            "skip": skip,
        }

    def __str__(self):
        formatted_str = f"{self.base_test_case.treatment_variable} _||_ {self.base_test_case.outcome_variable}"
        if self.adjustment_vars:
            formatted_str += f" | {self.adjustment_vars}"
        return formatted_str


def generate_metamorphic_relation(
    node_pair: tuple[str, str], dag: CausalDAG, nodes_to_ignore: set = None
) -> MetamorphicRelation:
    """Construct a metamorphic relation for a given node pair implied by the Causal DAG, or None if no such relation can
    be constructed (e.g. because every valid adjustment set contains a node to ignore).

    :param node_pair: The pair of nodes to consider.
    :param dag: Causal DAG from which the metamorphic relations will be generated.
    :param nodes_to_ignore: Set of nodes which will be excluded from causal tests.

    :return: A list containing ShouldCause and ShouldNotCause metamorphic relations.
    """

    if nodes_to_ignore is None:
        nodes_to_ignore = set()

    (u, v) = node_pair
    metamorphic_relations = []

    # Create a ShouldNotCause relation for each pair of nodes that are not directly connected
    if ((u, v) not in dag.edges) and ((v, u) not in dag.edges):
        # Case 1: U --> ... --> V
        if u in nx.ancestors(dag.graph, v):
            adj_sets = dag.direct_effect_adjustment_sets([u], [v], nodes_to_ignore=nodes_to_ignore)
            if adj_sets:
                metamorphic_relations.append(ShouldNotCause(BaseTestCase(u, v), list(adj_sets[0])))

        # Case 2: V --> ... --> U
        elif v in nx.ancestors(dag.graph, u):
            adj_sets = dag.direct_effect_adjustment_sets([v], [u], nodes_to_ignore=nodes_to_ignore)
            if adj_sets:
                metamorphic_relations.append(ShouldNotCause(BaseTestCase(v, u), list(adj_sets[0])))

        # Case 3: V _||_ U (No directed walk from V to U but there may be a back-door path e.g. U <-- Z --> V).
        # Only make one MR since V _||_ U == U _||_ V
        else:
            adj_sets = dag.direct_effect_adjustment_sets([u], [v], nodes_to_ignore=nodes_to_ignore)
            if adj_sets:
                metamorphic_relations.append(ShouldNotCause(BaseTestCase(u, v), list(adj_sets[0])))

    # Create a ShouldCause relation for each edge (u, v) or (v, u)
    elif (u, v) in dag.edges:
        adj_sets = dag.direct_effect_adjustment_sets([u], [v], nodes_to_ignore=nodes_to_ignore)
        if adj_sets:
            metamorphic_relations.append(ShouldCause(BaseTestCase(u, v), list(adj_sets[0])))
    else:
        adj_sets = dag.direct_effect_adjustment_sets([v], [u], nodes_to_ignore=nodes_to_ignore)
        if adj_sets:
            metamorphic_relations.append(ShouldCause(BaseTestCase(v, u), list(adj_sets[0])))
    return metamorphic_relations


def generate_metamorphic_relations(
    dag: CausalDAG, nodes_to_ignore: set = None, threads: int = 0, nodes_to_test: set = None
) -> list[MetamorphicRelation]:
    """Construct a list of metamorphic relations implied by the Causal DAG.

    This list of metamorphic relations contains a ShouldCause relation for every edge, and a ShouldNotCause
    relation for every (minimal) conditional independence relation implied by the structure of the DAG.

    :param dag: Causal DAG from which the metamorphic relations will be generated.
    :param nodes_to_ignore: Set of nodes which will be excluded from causal tests.
    :param threads: Number of threads to use (if generating in parallel).
    :param nodes_to_test: Set of nodes to test the relationships between (defaults to all nodes).

    :return: A list containing ShouldCause and ShouldNotCause metamorphic relations.
    """

    if nodes_to_ignore is None:
        nodes_to_ignore = {}

    if nodes_to_test is None:
        nodes_to_test = dag.nodes

    if threads < 2:
        metamorphic_relations = [
            generate_metamorphic_relation(node_pair, dag, nodes_to_ignore)
            for node_pair in combinations(filter(lambda node: node not in nodes_to_ignore, nodes_to_test), 2)
        ]
    else:
        with Pool(threads) as pool:
            metamorphic_relations = pool.starmap(
                generate_metamorphic_relation,
                map(
                    lambda node_pair: (node_pair, dag, nodes_to_ignore),
                    combinations(filter(lambda node: node not in nodes_to_ignore, nodes_to_test), 2),
                ),
            )

    return [item for items in metamorphic_relations for item in items]


def generate_causal_tests(dag_path: str, output_path: str, ignore_cycles: bool = False, threads: int = 0):
    """
    Generate and output causal tests for a given DAG.

    :param dag_path: Path to the DOT file that specifies the causal DAG.
    :param output_path: Path to save the JSON output.
    :param ignore_cycles: Whether to bypass the check that the DAG is actually acyclic. If set to true, tests that
                          include variables that are part of a cycle as either treatment, outcome, or adjustment will
                          be omitted from the test set.
    :param threads: The number of threads to use to generate tests in parallel. If unspecified, tests are generated in
                    serial. This is tylically fine unless the number of tests to be generated is >10000.
    """
    causal_dag = CausalDAG(dag_path, ignore_cycles=ignore_cycles)

    dag_nodes_to_test = [
        node for node in causal_dag.nodes if nx.get_node_attributes(causal_dag.graph, "test", default=True)[node]
    ]

    if not causal_dag.is_acyclic() and ignore_cycles:
        logger.warning(
            "Ignoring cycles by removing causal tests that reference any node within a cycle. "
            "Your causal test suite WILL NOT BE COMPLETE!"
        )
        relations = generate_metamorphic_relations(
            causal_dag,
            nodes_to_test=dag_nodes_to_test,
            nodes_to_ignore=set(causal_dag.cycle_nodes()),
            threads=threads,
        )
    else:
        relations = generate_metamorphic_relations(causal_dag, nodes_to_test=dag_nodes_to_test, threads=threads)

    tests = [
        relation.to_json_stub(skip=False)
        for relation in relations
        if len(list(causal_dag.graph.predecessors(relation.base_test_case.outcome_variable))) > 0
    ]

    logger.info(f"Generated {len(tests)} tests. Saving to {output_path}.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tests": tests}, f, indent=2)
