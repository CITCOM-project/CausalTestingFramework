"""
This module contains code to measure various aspects of causal test adequacy.
"""

import logging
from itertools import combinations

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_case import CausalTestCase

logger = logging.getLogger(__name__)


class DAGAdequacy:
    """
    Measures the adequacy of a given DAG by hos many edges and independences are tested.
    """

    def __init__(
        self,
        causal_dag: CausalDAG,
        test_suite: list[CausalTestCase],
    ):
        self.causal_dag = causal_dag
        self.test_suite = test_suite
        self.tested_pairs = None
        self.pairs_to_test = None
        self.untested_pairs = None
        self.dag_adequacy = None

    def measure_adequacy(self):
        """
        Calculate the adequacy measurement, and populate the `dag_adequacy` field.
        """
        self.pairs_to_test = set(combinations(self.causal_dag.nodes, 2))
        self.tested_pairs = set()

        for n1, n2 in self.pairs_to_test:
            if (n1, n2) in self.causal_dag.edges():
                if any((t.treatment_variable, t.outcome_variable) == (n1, n2) for t in self.test_suite):
                    self.tested_pairs.add((n1, n2))
            else:
                # Causal independences are not order dependent
                if any((t.treatment_variable, t.outcome_variable) in {(n1, n2), (n2, n1)} for t in self.test_suite):
                    self.tested_pairs.add((n1, n2))

        self.untested_pairs = self.pairs_to_test.difference(self.tested_pairs)
        self.dag_adequacy = len(self.tested_pairs) / len(self.pairs_to_test)

    def to_dict(self):
        """Returns the adequacy object as a dictionary."""
        return {
            "causal_dag": self.causal_dag,
            "test_suite": self.test_suite,
            "tested_pairs": self.tested_pairs,
            "pairs_to_test": self.pairs_to_test,
            "untested_pairs": self.untested_pairs,
            "dag_adequacy": self.dag_adequacy,
        }
