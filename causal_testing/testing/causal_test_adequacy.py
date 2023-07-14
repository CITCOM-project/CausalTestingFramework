"""
This module contains code to measure various aspects of causal test adequacy.
"""
from causal_testing.testing.causal_test_suite import CausalTestSuite
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import Estimator
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_engine import CausalTestEngine
from itertools import combinations
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.model_selection import cross_val_score


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
    def __init__(self, test_case: CausalTestCase, test_engine: CausalTestEngine, estimator: Estimator):
        self.test_case = test_case
        self.test_engine = test_engine
        self.estimator = estimator

    def measure_adequacy_bootstrap(self, bootstrap_size: int = 100):
        results = []
        for i in range(bootstrap_size):
            estimator = deepcopy(self.estimator)
            estimator.df = estimator.df.sample(len(estimator.df), replace=True, random_state=i)
            results.append(self.test_engine.execute_test(estimator, self.test_case))
        return results

    def measure_adequacy_k_folds(self, k: int = 10, random_state=0):
        results = []
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_inx, test_inx in kf.split(self.estimator.df):
            estimator = deepcopy(self.estimator)
            test = estimator.df.iloc[test_inx]
            estimator.df = estimator.df.iloc[train_inx]
            test_result = estimator.model.predict(test)
            results.append(np.sqrt(mse(test_result, test[self.test_case.base_test_case.outcome_variable.name])).mean())
        return np.mean(results)
