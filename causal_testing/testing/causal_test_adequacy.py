"""
This module contains code to measure various aspects of causal test adequacy.
"""

import logging
from copy import deepcopy
from itertools import combinations

import pandas as pd

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


class DataAdequacy:
    """
    Measures the adequacy of a given test according to the Fisher kurtosis of the bootstrapped result.

    * Positive kurtoses indicate the model doesn't have enough data so is unstable.
    * Negative kurtoses indicate the model doesn't have enough data, but is too stable,
      indicating that the spread of inputs is insufficient.
    * Zero kurtosis is optimal.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        test_case: CausalTestCase,
        bootstrap_size: int = 100,
        group_by=None,
    ):
        self.test_case = test_case
        self.kurtosis = None
        self.passing = None
        self.results = None
        self.successful = None
        self.bootstrap_size = bootstrap_size
        self.group_by = group_by

    def measure_adequacy(self):
        """
        Calculate the adequacy measurement, and populate the data_adequacy field.
        """
        results = []
        outcomes = []
        for i in range(self.bootstrap_size):
            estimator = deepcopy(self.test_case.estimator)

            if self.group_by is not None:
                ids = pd.Series(estimator.df[self.group_by].unique())
                ids = ids.sample(len(ids), replace=True, random_state=i)
                estimator.df = estimator.df[estimator.df[self.group_by].isin(ids)]
            else:
                estimator.df = estimator.df.sample(len(estimator.df), replace=True, random_state=i)
            result = self.test_case.execute_test(estimator)
            outcomes.append(self.test_case.expected_causal_effect.apply(result))
            results.append(result.effect_estimate.to_df())
        results = pd.concat(results)
        results["var"] = results.index
        results["passed"] = outcomes

        self.results = results
        self.kurtosis = results.groupby("var")["effect_estimate"].apply(lambda x: x.kurtosis())
        self.passing = sum(filter(lambda x: x is not None, outcomes))
        self.successful = sum(x is not None for x in outcomes)

    def to_dict(self):
        """Returns the adequacy object as a dictionary."""
        return {
            "kurtosis": self.kurtosis.to_dict(),
            "bootstrap_size": self.bootstrap_size,
            "passing": self.passing,
            "successful": self.successful,
            "results": self.results.reset_index(drop=True).to_dict(),
        }
