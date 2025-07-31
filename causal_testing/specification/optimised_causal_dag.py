"""This module contains the CausalDAG class, as well as the functions list_all_min_sep and close_seperator"""

from __future__ import annotations

import logging
from itertools import combinations
from random import sample
from typing import Union, Generator, Set

import networkx as nx

from causal_testing.specification.causal_dag import CausalDAG

Node = Union[str, int]  # Node type hint: A node is a string or an int

logger = logging.getLogger(__name__)


class OptimisedCausalDAG(CausalDAG):

    def enumerate_minimal_adjustment_sets(self, treatments: list[str], outcomes: list[str]) -> Generator[set[str]]:
        """Get the smallest possible set of variables that blocks all back-door paths between all pairs of treatments
        and outcomes.

        This is an implementation of the Algorithm presented in Adjustment Criteria in Causal Diagrams: An
        Algorithmic Perspective, Textor and Lískiewicz, 2012 and extended in Separators and adjustment sets in causal
        graphs: Complete criteria and an algorithmic framework, Zander et al.,  2019. These works use the algorithm
        presented by Takata et al. in their work entitled: Space-optimal, backtracking algorithms to list the minimal
        vertex separators of a graph, 2013.

        At a high-level, this algorithm proceeds as follows for a causal DAG G, set of treatments X, and set of
        outcomes Y):
        1). Transform G to a proper back-door graph G_pbd (remove the first edge from X on all proper causal paths).
        2). Transform G_pbd to the ancestor moral graph (G_pbd[An(X union Y)])^m.
        3). Apply Takata's algorithm to output all minimal X-Y separators in the graph.

        :param treatments: A list of strings representing treatments.
        :param outcomes: A list of strings representing outcomes.
        :return: A list of strings representing the minimal adjustment set.
        """

        # Step 1: Build the proper back-door graph and its moralized ancestor graph
        proper_backdoor_graph = self.get_proper_backdoor_graph(treatments, outcomes)
        ancestor_proper_backdoor_graph = proper_backdoor_graph.get_ancestor_graph(treatments, outcomes)
        moralised_proper_backdoor_graph = nx.moral_graph(ancestor_proper_backdoor_graph.graph)

        # Step 2: Add artificial TREATMENT and OUTCOME nodes
        moralised_proper_backdoor_graph.add_edges_from([("TREATMENT", t) for t in treatments])
        moralised_proper_backdoor_graph.add_edges_from([("OUTCOME", y) for y in outcomes])

        # Step 3: Remove treatment and outcome nodes from graph and connect neighbours
        treatment_neighbors = {
            node for t in treatments for node in moralised_proper_backdoor_graph[t] if node not in treatments
        }
        moralised_proper_backdoor_graph.add_edges_from(combinations(treatment_neighbors, 2))

        outcome_neighbors = {
            node for o in outcomes for node in moralised_proper_backdoor_graph[o] if node not in outcomes
        }
        moralised_proper_backdoor_graph.add_edges_from(combinations(outcome_neighbors, 2))

        # Step 4: Find all minimal separators of X^m and Y^m using Takata's algorithm for listing minimal separators
        sep_candidates = self.list_all_min_sep_opt(
            moralised_proper_backdoor_graph,
            "TREATMENT",
            "OUTCOME",
            {"TREATMENT"},
            set(moralised_proper_backdoor_graph["OUTCOME"]) | {"OUTCOME"},
        )
        return filter(
            lambda s: self.constructive_backdoor_criterion(proper_backdoor_graph, treatments, outcomes, s),
            sep_candidates,
        )

    def constructive_backdoor_criterion(
        self,
        proper_backdoor_graph: CausalDAG,
        treatments: list[str],
        outcomes: list[str],
        covariates: list[str],
    ) -> bool:
        """A variation of Pearl's back-door criterion applied to a proper backdoor graph which enables more efficient
        computation of minimal adjustment sets for the effect of a set of treatments on a set of outcomes.

        The constructive back-door criterion is satisfied for a causal DAG G, a set of treatments X, a set of outcomes
        Y, and a set of covariates Z, if:
        (1) Z is not a descendent of any variable on a proper causal path between X and Y.
        (2) Z d-separates X and Y in the proper back-door graph relative to X and Y.

        Reference: (Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework,
        Zander et al.,  2019, Definition 4, p.16)

        :param proper_backdoor_graph: A proper back-door graph relative to the specified treatments and outcomes.
        :param treatments: A list of treatment variables that appear in the proper back-door graph.
        :param outcomes: A list of outcome variables that appear in the proper back-door graph.
        :param covariates: A list of variables that appear in the proper back-door graph that we will check against
        the constructive back-door criterion.
        :return: True or False, depending on whether the set of covariates satisfies the constructive back-door
        criterion.
        """

        # Condition (1): Covariates must not be descendants of any node on a proper causal path
        proper_path_vars = self.proper_causal_pathway(treatments, outcomes)
        if proper_path_vars:
            # Collect all descendants including each proper causal path var itself
            descendents_of_proper_casual_paths = set(proper_path_vars).union(
                {node for var in proper_path_vars for node in nx.descendants(self.graph, var)}
            )

            if not set(covariates).issubset(set(self.nodes).difference(descendents_of_proper_casual_paths)):
                # Covariates intersect with disallowed descendants — fail condition 1
                logger.info(
                    "Failed Condition 1: "
                    "Z=%s **is** a descendant of variables on a proper causal path between X=%s and Y=%s.",
                    covariates,
                    treatments,
                    outcomes,
                )
                return False

        # Condition (2): Z must d-separate X and Y in the proper back-door graph
        if not nx.d_separated(proper_backdoor_graph.graph, set(treatments), set(outcomes), set(covariates)):
            logger.info(
                "Failed Condition 2: Z=%s **does not** d-separate X=%s and Y=%s in the proper back-door graph.",
                covariates,
                treatments,
                outcomes,
            )
            return False

        return True

    def list_all_min_sep_opt(
        self,
        graph: nx.Graph,
        treatment_node: str,
        outcome_node: str,
        treatment_node_set: Set,
        outcome_node_set: Set,
    ) -> Generator[Set, None, None]:
        """A backtracking algorithm for listing all minimal treatment-outcome separators in an undirected graph.

        Reference: (Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph, Ken Takata,
        2013, p.5, ListMinSep procedure).

        :param graph: An undirected graph.
        :param treatment_node: The node corresponding to the treatment variable we wish to separate from the output.
        :param outcome_node: The node corresponding to the outcome variable we wish to separate from the input.
        :param treatment_node_set: Set of treatment nodes.
        :param outcome_node_set: Set of outcome nodes.
        :return: A generator of minimal-sized sets of variables which separate treatment and outcome in the undirected
                 graph.
        """
        # 1. Compute the close separator of the treatment set
        close_separator_set = self.close_separator_opt(graph, treatment_node, outcome_node, treatment_node_set)

        # 2. Use the close separator to separate the graph and obtain the connected components (connected sub-graphs)
        components_graph = graph.subgraph(set(graph.nodes) - close_separator_set)

        # 3. Find the component containing the treatment node
        treatment_component = set()
        for component in nx.connected_components(components_graph):
            if treatment_node in component:
                treatment_component = component
                break

        # 4. Confirm that the connected component containing the treatment node is disjoint with the outcome node set
        if treatment_component.intersection(outcome_node_set):
            return

        # 5. Update the treatment node set to the set of nodes in the connected component containing the treatment node
        treatment_node_set = treatment_component

        # 6. Obtain the neighbours of the new treatment node set (this excludes the treatment nodes themselves)
        neighbour_nodes = {
            neighbour for node in treatment_node_set for neighbour in graph[node] if neighbour not in treatment_node_set
        }

        # 7. Check that there exists at least one neighbour of the treatment nodes that is not in the outcome node set
        remaining = neighbour_nodes - outcome_node_set
        if remaining:
            # 7.1. If so, sample a random node from the set of treatment nodes' neighbours not in the outcome node set
            chosen = sample(sorted(remaining), 1)
            # 7.2. Add this node to the treatment node set and recurse (left branch)
            yield from self.list_all_min_sep_opt(
                graph,
                treatment_node,
                outcome_node,
                treatment_node_set.union(chosen),
                outcome_node_set,
            )
            # 7.3. Add this node to the outcome node set and recurse (right branch)
            yield from self.list_all_min_sep_opt(
                graph,
                treatment_node,
                outcome_node,
                treatment_node_set,
                outcome_node_set.union(chosen),
            )
        else:
            # Step 8: All neighbours are in outcome set — we found a separator
            yield neighbour_nodes

    def close_separator_opt(
        self, graph: nx.Graph, treatment_node: Node, outcome_node: Node, treatment_node_set: set[Node]
    ) -> set[Node]:
        """Compute the close separator for a set of treatments in an undirected graph.

        A close separator (relative to a set of variables X) is a separator whose vertices are adjacent to those in X.
        An X-Y separator is a set of variables which, once deleted from a graph, create a subgraph in which X and Y
        are in different components.

        Reference: (Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph, Ken Takata,
        2013, p.4, CloseSeparator procedure).

        :param graph: An undirected graph.
        :param treatment_node: A label for the treatment node (parent of treatments in undirected graph).
        :param outcome_node: A label for the outcome node (parent of outcomes in undirected graph).
        :param treatment_node_set: The set of variables containing the treatment node ({treatment_node}).
        :return: A treatment_node-outcome_node separator whose vertices are adjacent to those in treatments.
        """
        treatment_neighbours = set.union(*[set(nx.neighbors(graph, treatment)) for treatment in treatment_node_set])
        components_graph = graph.subgraph(set(graph.nodes) - treatment_neighbours)
        graph_components = nx.connected_components(components_graph)
        for component in graph_components:
            if outcome_node in component:
                neighbours_of_variables_in_component = set.union(
                    *[set(nx.neighbors(graph, variable)) for variable in component]
                )
                # For this algorithm, the neighbours of a node do not include the node itself
                neighbours_of_variables_in_component = neighbours_of_variables_in_component.difference(component)
                return neighbours_of_variables_in_component
        raise ValueError(f"No {treatment_node}-{outcome_node} separator in the graph.")
