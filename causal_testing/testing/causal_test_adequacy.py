"""
This module contains code to measure various aspects of causal test adequacy.
"""
from causal_testing.testing.causal_test_suite import CausalTestSuite
from causal_testing.data_collection.data_collector import DataCollector
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.causal_test_case import CausalTestCase
from itertools import combinations
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd


class DAGAdequacy:
    def __init__(
        self,
        causal_specification: CausalSpecification,
        test_suite: CausalTestSuite,
    ):
        self.causal_dag = causal_specification.causal_dag
        self.test_suite = test_suite

    def measure_adequacy(self):
        self.tested_pairs = {
            (t.base_test_case.treatment_variable, t.base_test_case.outcome_variable) for t in self.causal_test_suite
        }
        self.pairs_to_test = set(combinations(self.causal_dag.graph.nodes, 2))
        self.untested_edges = pairs_to_test.difference(tested_pairs)
        self.dag_adequacy = len(tested_pairs) / len(pairs_to_test)


class DataAdequacy:
    def __init__(
        self, test_case: CausalTestCase, estimator: Estimator, data_collector: DataCollector, bootstrap_size: int = 100
    ):
        self.test_case = test_case
        self.estimator = estimator
        self.data_collector = data_collector
        self.kurtosis = None
        self.outcomes = None
        self.bootstrap_size = bootstrap_size

    def measure_adequacy(self):
        results = []
        for i in range(self.bootstrap_size):
            estimator = deepcopy(self.estimator)
            estimator.df = estimator.df.sample(len(estimator.df), replace=True, random_state=i)
            try:
                results.append(self.test.execute_test(estimator, self.data_collector))
            except np.LinAlgError:
                continue
        outcomes = [self.test_case.expected_causal_effect.apply(c) for c in results]
        results = pd.DataFrame(c.to_dict() for c in results)[["effect_estimate", "ci_low", "ci_high"]]

        def convert_to_df(field):
            converted = []
            for r in results[field]:
                if isinstance(r, float):
                    converted.append(
                        pd.DataFrame({self.test_case.base_test_case.treatment_variable.name: [r]}).transpose()
                    )
                else:
                    converted.append(r)
            return converted

        for field in ["effect_estimate", "ci_low", "ci_high"]:
            results[field] = convert_to_df(field)

        effect_estimate = pd.concat(results["effect_estimate"].tolist(), axis=1).transpose().reset_index(drop=True)
        self.kurtosis = effect_estimate.kurtosis()
        self.outcomes = sum(outcomes)

    def to_dict(self):
        return {"kurtosis": self.kurtosis.to_dict(), "bootstrap_size": self.bootstrap_size, "passing": self.outcomes}
